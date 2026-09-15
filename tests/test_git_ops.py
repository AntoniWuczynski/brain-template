"""Tests for mcp_server.git_ops.

Three groups: the ``expected_branch`` guard (AUD-003 /
review/2026-09-03/findings-dispatcher.md F3) that refuses to commit onto
a branch other than the one configured, and the remaining outcome
branches — no paths, a no-op commit, a path outside the vault, a commit
that fails, and the subprocess timeout (AUD-065). The no-op branch is the
one every idempotent write hits, so a regression there turns each
repeated write into a GitError although the file is on disk — and
``pull_rebase`` (AUD-122), driven through two clones of one bare repo:
the server's vault and the Obsidian clone that also pushes to it.

The throwaway git repo comes from the shared ``mcp_vault`` / ``run_git``
fixtures in tests/conftest.py.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NoReturn, TYPE_CHECKING

import pytest

# Type-only import of the shared harness contracts. At runtime the module
# is pytest's ``conftest``; ``tests.conftest`` is the name mypy resolves
# (explicit_package_bases), and importing it for real would collide with an
# unrelated installed ``tests`` package. Annotations are postponed, so
# nothing here is evaluated at import time.
if TYPE_CHECKING:
    from tests.conftest import GitRunner

from mcp_server import git_ops
from mcp_server.git_ops import (
    DirtyTreeError,
    GitError,
    RebaseConflictError,
    RebaseOutcome,
    commit_paths,
    pull_rebase,
)


def _seed(repo: Path, run_git: GitRunner) -> None:
    """One commit, so the repo has a HEAD and can grow branches."""
    (repo / "seed.md").write_text("seed\n", encoding="utf-8")
    run_git(repo, "add", "seed.md")
    run_git(repo, "commit", "-q", "-m", "seed")


def _log_subjects(repo: Path, run_git: GitRunner) -> list[str]:
    out = run_git(repo, "log", "--format=%s")
    return out.splitlines() if out else []


# ------------------------------------------------------- expected_branch

def test_commits_when_expected_branch_matches(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    note = mcp_vault / "note.md"
    note.write_text("hello\n", encoding="utf-8")

    outcome = commit_paths(
        mcp_vault, paths=[note], message="add note", expected_branch="main"
    )

    assert outcome.committed is True
    assert outcome.sha is not None
    assert "add note" in run_git(mcp_vault, "log", "--oneline")


def test_refuses_to_commit_on_wrong_branch(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    _seed(mcp_vault, run_git)
    run_git(mcp_vault, "checkout", "-q", "-b", "side")
    note = mcp_vault / "note.md"
    note.write_text("hello\n", encoding="utf-8")

    with pytest.raises(GitError, match="side"):
        commit_paths(mcp_vault, paths=[note], message="add note", expected_branch="main")

    assert "?? note.md" in run_git(mcp_vault, "status", "--porcelain")


def test_expected_branch_none_keeps_old_behaviour(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    _seed(mcp_vault, run_git)
    run_git(mcp_vault, "checkout", "-q", "-b", "side")
    note = mcp_vault / "note.md"
    note.write_text("hello\n", encoding="utf-8")

    outcome = commit_paths(mcp_vault, paths=[note], message="add note", expected_branch=None)

    assert outcome.committed is True
    assert outcome.sha is not None


def test_refuses_to_commit_on_detached_head(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    _seed(mcp_vault, run_git)
    run_git(mcp_vault, "checkout", "-q", "--detach")
    note = mcp_vault / "note.md"
    note.write_text("hello\n", encoding="utf-8")

    with pytest.raises(GitError, match="detached HEAD"):
        commit_paths(mcp_vault, paths=[note], message="add note", expected_branch="main")

    assert "?? note.md" in run_git(mcp_vault, "status", "--porcelain")


def test_current_branch_names_the_branch_or_detached(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    assert git_ops.current_branch(mcp_vault) == "main"
    _seed(mcp_vault, run_git)
    run_git(mcp_vault, "checkout", "-q", "-b", "side")
    assert git_ops.current_branch(mcp_vault) == "side"
    run_git(mcp_vault, "checkout", "-q", "--detach")
    assert git_ops.current_branch(mcp_vault) == "(detached HEAD)"


# ------------------------------------------------- the other outcome branches

def test_no_paths_returns_a_noop_outcome(mcp_vault: Path, run_git: GitRunner) -> None:
    # Returns before the lock and before any git call at all — an empty
    # path list is not an error, it is nothing to do.
    _seed(mcp_vault, run_git)

    outcome = commit_paths(mcp_vault, paths=[], message="nothing", expected_branch="main")

    assert outcome == git_ops.CommitOutcome(None, False, False, "no paths given")
    assert _log_subjects(mcp_vault, run_git) == ["seed"]


def test_recommitting_identical_content_is_a_noop(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    # The idempotency branch (AGENTS.md rule 7): appending identical content
    # or re-upserting an existing relation reaches git with nothing staged.
    # It must report the no-op, not raise the `git commit` exit-1 failure.
    note = mcp_vault / "note.md"
    note.write_text("hello\n", encoding="utf-8")
    first = commit_paths(mcp_vault, paths=[note], message="add note", expected_branch="main")
    assert first.committed is True

    note.write_text("hello\n", encoding="utf-8")   # byte-identical rewrite
    second = commit_paths(
        mcp_vault, paths=[note], message="add note again", expected_branch="main"
    )

    assert second.committed is False
    assert second.sha is None
    assert second.pushed is False
    assert second.detail == "no changes to commit"
    assert _log_subjects(mcp_vault, run_git) == ["add note"]   # no empty commit


def test_path_outside_the_vault_is_refused(
    mcp_vault: Path, tmp_path: Path, run_git: GitRunner
) -> None:
    _seed(mcp_vault, run_git)
    outside = tmp_path.resolve() / "elsewhere.md"
    outside.write_text("not mine\n", encoding="utf-8")

    with pytest.raises(GitError, match="not inside the vault"):
        commit_paths(
            mcp_vault, paths=[outside], message="steal", expected_branch="main"
        )

    assert _log_subjects(mcp_vault, run_git) == ["seed"]
    # The refusal happens before anything is staged.
    assert run_git(mcp_vault, "diff", "--cached", "--name-only") == ""


def test_failing_commit_surfaces_as_a_git_error(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    # A commit can fail for reasons the caller can't predict (a hook, a
    # locked index, a full disk). The failure must reach the write tool as
    # GitError — not as a silent success.
    _seed(mcp_vault, run_git)
    hook = mcp_vault / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'hook says no' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    note = mcp_vault / "note.md"
    note.write_text("hello\n", encoding="utf-8")

    with pytest.raises(GitError) as exc:
        commit_paths(mcp_vault, paths=[note], message="add note", expected_branch="main")

    assert "git commit" in str(exc.value)
    assert "hook says no" in str(exc.value)
    assert _log_subjects(mcp_vault, run_git) == ["seed"]


def test_timeout_becomes_a_git_error(
    mcp_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The timeout exists so a black-holed remote can't hang a push while
    # holding _GIT_LOCK and wedging every write tool.
    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise subprocess.TimeoutExpired(cmd=["git", "push"], timeout=30.0)

    # git_ops calls subprocess.run through the same module object.
    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(GitError, match=r"git push timed out after 30\.0s"):
        git_ops._git(mcp_vault, "push", "origin", "main")


# ------------------------------------------------------------ pull_rebase
#
# AUD-122: after the migration the server's clone and the owner's Obsidian
# clone both write to one remote, so every write has to absorb the other
# side first. Each test drives two real clones of one bare repo.


def _identity(repo: Path, run_git: GitRunner) -> None:
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "test")


def _clone_pair(tmp_path: Path, run_git: GitRunner) -> tuple[Path, Path]:
    """``(server, obsidian)`` — two clones of one bare remote, both holding
    a seed commit with ``note.md`` in it."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)

    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    _identity(seed, run_git)
    (seed / "note.md").write_text("seed\n", encoding="utf-8")
    run_git(seed, "add", "note.md")
    run_git(seed, "commit", "-q", "-m", "seed")
    run_git(seed, "remote", "add", "origin", str(bare))
    run_git(seed, "push", "-q", "origin", "main")

    clones: list[Path] = []
    for name in ("server", "obsidian"):
        clone = tmp_path / name
        subprocess.run(["git", "clone", "-q", str(bare), str(clone)], check=True)
        _identity(clone, run_git)
        clones.append(clone)
    return clones[0], clones[1]


def _commit_file(repo: Path, name: str, text: str, run_git: GitRunner) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    run_git(repo, "add", name)
    run_git(repo, "commit", "-q", "-m", name)
    return run_git(repo, "rev-parse", "HEAD")


def test_pull_rebase_fast_forwards_when_the_clone_has_no_commits(
    tmp_path: Path, run_git: GitRunner
) -> None:
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")

    outcome = pull_rebase(server, remote="origin", branch="main")

    assert outcome == RebaseOutcome(fetched=True, absorbed=1, rebased=False, detail=None)
    assert (server / "hand.md").read_text(encoding="utf-8") == "typed by hand\n"


def test_pull_rebase_replays_local_commits_on_top_of_the_remote(
    tmp_path: Path, run_git: GitRunner
) -> None:
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    _commit_file(server, "agent.md", "written by an agent\n", run_git)

    outcome = pull_rebase(server, remote="origin", branch="main")

    assert outcome.rebased is True
    assert outcome.absorbed == 1
    # The agent's commit now sits ON TOP of the hand edit, and both files
    # are present — that is the acceptance criterion in AUD-122.
    assert _log_subjects(server, run_git)[:2] == ["agent.md", "hand.md"]
    assert (server / "hand.md").exists()
    assert (server / "agent.md").exists()


def test_pull_rebase_refuses_a_conflict_and_leaves_the_tree_clean(
    tmp_path: Path, run_git: GitRunner
) -> None:
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "note.md", "edited by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    head_before = _commit_file(server, "note.md", "edited by the agent\n", run_git)

    with pytest.raises(RebaseConflictError) as excinfo:
        pull_rebase(server, remote="origin", branch="main")

    assert excinfo.value.paths == ("note.md",)
    assert "note.md" in str(excinfo.value)
    # Nothing resolved in either direction (IMP-029): same HEAD, same file,
    # clean tree, no rebase left in progress.
    assert run_git(server, "rev-parse", "HEAD") == head_before
    assert run_git(server, "status", "--porcelain") == ""
    assert (server / "note.md").read_text(encoding="utf-8") == "edited by the agent\n"
    assert not (server / ".git" / "rebase-merge").exists()
    assert not (server / ".git" / "rebase-apply").exists()


def test_pull_rebase_is_a_no_op_when_already_up_to_date(
    tmp_path: Path, run_git: GitRunner
) -> None:
    server, _obsidian = _clone_pair(tmp_path, run_git)
    head_before = run_git(server, "rev-parse", "HEAD")

    outcome = pull_rebase(server, remote="origin", branch="main")

    assert outcome == RebaseOutcome(fetched=True, absorbed=0, rebased=False, detail=None)
    assert run_git(server, "rev-parse", "HEAD") == head_before


def test_pull_rebase_reports_an_unreachable_remote_without_raising(
    tmp_path: Path, run_git: GitRunner
) -> None:
    """The laptop case: offline must not fail the write that follows."""
    server, _obsidian = _clone_pair(tmp_path, run_git)
    missing = tmp_path / "gone.git"
    run_git(server, "remote", "set-url", "origin", str(missing))
    head_before = run_git(server, "rev-parse", "HEAD")

    outcome = pull_rebase(server, remote="origin", branch="main")

    assert outcome.fetched is False
    assert outcome.absorbed == 0
    assert outcome.detail is not None
    assert str(missing) not in outcome.detail  # no remote paths in what callers show
    assert run_git(server, "rev-parse", "HEAD") == head_before


def test_pull_rebase_refuses_a_dirty_tree_naming_the_paths(
    tmp_path: Path, run_git: GitRunner
) -> None:
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    (server / "note.md").write_text("half-written\n", encoding="utf-8")

    with pytest.raises(DirtyTreeError) as excinfo:
        pull_rebase(server, remote="origin", branch="main")

    assert excinfo.value.paths == ("note.md",)
    assert "note.md" in str(excinfo.value)
    assert not (server / "hand.md").exists()  # refused before touching anything


def test_pull_rebase_ignores_untracked_files(
    tmp_path: Path, run_git: GitRunner
) -> None:
    """A real vault always has untracked files (an unprocessed inbox); they
    are not a half-done write and must not refuse every rebase."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    (server / "scratch.txt").write_text("not in git\n", encoding="utf-8")

    outcome = pull_rebase(server, remote="origin", branch="main")

    assert outcome.absorbed == 1
    assert (server / "scratch.txt").exists()


def test_pull_rebase_refuses_a_branch_other_than_the_configured_one(
    tmp_path: Path, run_git: GitRunner
) -> None:
    """Review 2026-09-09 #4: ``pull_rebase`` had no branch guard, so the
    push worker rebased whatever HEAD pointed at — rewriting an unrelated
    branch's history while leaving the branch it meant to fix untouched."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    run_git(server, "checkout", "-q", "-b", "sidebar")
    side_head = _commit_file(server, "side.md", "side work\n", run_git)
    main_head = run_git(server, "rev-parse", "main")

    with pytest.raises(GitError, match="sidebar"):
        pull_rebase(server, remote="origin", branch="main", expected_branch="main")

    assert run_git(server, "rev-parse", "sidebar") == side_head
    assert run_git(server, "rev-parse", "main") == main_head
    assert not (server / "hand.md").exists()


def test_pull_rebase_refuses_a_detached_head(
    tmp_path: Path, run_git: GitRunner
) -> None:
    """The other half of #4: on a detached HEAD the fast-forward advanced
    HEAD and left the configured branch behind."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    run_git(server, "checkout", "-q", "--detach")
    head_before = run_git(server, "rev-parse", "HEAD")

    with pytest.raises(GitError, match="detached HEAD"):
        pull_rebase(server, remote="origin", branch="main", expected_branch="main")

    assert run_git(server, "rev-parse", "HEAD") == head_before
    assert not (server / "hand.md").exists()


def test_pull_rebase_reports_a_blocked_fast_forward_instead_of_raising(
    tmp_path: Path, run_git: GitRunner
) -> None:
    """Review 2026-09-09 #5: an untracked file where an incoming commit
    adds one makes ``merge --ff-only`` fail. That used to escape as a bare
    GitError past every caller; it is an outcome, not an exception — the
    write that follows must still land locally."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    _commit_file(obsidian, "collide.md", "hand version\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    (server / "collide.md").write_text("server scratch\n", encoding="utf-8")
    head_before = run_git(server, "rev-parse", "HEAD")

    outcome = pull_rebase(server, remote="origin", branch="main", expected_branch="main")

    assert outcome.fetched is True
    assert outcome.absorbed == 0
    assert outcome.blocked is not None
    assert "collide.md" in outcome.blocked
    assert run_git(server, "rev-parse", "HEAD") == head_before
    assert (server / "collide.md").read_text(encoding="utf-8") == "server scratch\n"


def test_rebase_in_progress_sees_a_half_finished_rebase(
    tmp_path: Path, run_git: GitRunner
) -> None:
    """Review 2026-09-09 #6: a killed rebase leaves a CLEAN tree on a
    detached HEAD — no other check explains it."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    assert git_ops.rebase_in_progress(server) is False
    _commit_file(obsidian, "hand.md", "typed by hand\n", run_git)
    run_git(obsidian, "push", "-q", "origin", "main")
    _commit_file(server, "one.md", "one\n", run_git)
    _commit_file(server, "two.md", "two\n", run_git)
    run_git(server, "fetch", "-q", "origin", "main")
    subprocess.run(["git", "-C", str(server), "rebase", "--exec", "false", "FETCH_HEAD"],
                   capture_output=True, text=True, check=False)

    assert run_git(server, "status", "--porcelain", "--untracked-files=no") == ""
    assert git_ops.rebase_in_progress(server) is True
    subprocess.run(["git", "-C", str(server), "rebase", "--abort"],
                   capture_output=True, text=True, check=False)
    assert git_ops.rebase_in_progress(server) is False


def test_head_sha_reads_head_and_survives_an_empty_repo(
    tmp_path: Path, mcp_vault: Path, run_git: GitRunner
) -> None:
    assert git_ops.head_sha(mcp_vault) is None   # no commits yet
    _seed(mcp_vault, run_git)
    assert git_ops.head_sha(mcp_vault) == run_git(mcp_vault, "rev-parse", "HEAD")
