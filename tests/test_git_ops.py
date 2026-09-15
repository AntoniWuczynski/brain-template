"""Tests for mcp_server.git_ops.commit_paths.

Two groups: the ``expected_branch`` guard (AUD-003 /
review/2026-09-03/findings-dispatcher.md F3) that refuses to commit onto
a branch other than the one configured, and the remaining outcome
branches — no paths, a no-op commit, a path outside the vault, a commit
that fails, and the subprocess timeout (AUD-065). The no-op branch is the
one every idempotent write hits, so a regression there turns each
repeated write into a GitError although the file is on disk.

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
from mcp_server.git_ops import GitError, commit_paths


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
