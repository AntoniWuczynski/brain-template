"""Tests for mcp_server.sync — the write critical section and the
pull-before-write policy built on it (AUD-122, FOUNDER_DECISIONS IMP-029
and its 2026-09-09 correction).

Every scenario below is a defect the suite used to miss because it had no
test for concurrency, none for a conflict outside the note being written,
and none for HEAD moving mid-write. They drive the REAL write tools over
two clones of one bare repo — the server's vault and the Obsidian clone
that also pushes to it — because the bugs live in the interleaving of the
pull, the tool body and the commit, not in any one of them.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import CfgFactory, GitRunner

from mcp_server import sync, tools as tools_mod
from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.errors import ToolError
from mcp_server.git_ops import CommitOutcome, pull_rebase
from mcp_server.identity import AGENT_VAR
from mcp_server.provenance import stamp_provenance as real_stamp
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime

pytestmark = pytest.mark.usefixtures("roomy_rate_buckets")


# --------------------------------------------------------------- harness

def _identity(repo: Path, run_git: GitRunner) -> None:
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "test")


def _clone_pair(tmp_path: Path, run_git: GitRunner) -> tuple[Path, Path]:
    """``(server, obsidian)`` — two clones of one bare remote, seeded with
    a note each test can edit from both sides."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)

    seed = tmp_path / "seed"
    (seed / "knowledge" / "notes").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    _identity(seed, run_git)
    for name in ("shared", "other"):
        (seed / "knowledge" / "notes" / f"{name}.md").write_text(
            f"---\ntitle: {name}\n---\n\nseed body\n", encoding="utf-8"
        )
    run_git(seed, "add", "-A")
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


def _pushing_runtime(root: Path) -> Runtime:
    """A Runtime whose pull-before-write is live (``push_enabled``) but
    whose background push thread is stopped, so the test controls exactly
    what reaches the remote. Built before any local commit exists, so the
    one flush push ``stop()`` makes is a no-op."""
    audit = AuditLog(root)
    worker = PushWorker(root, remote="origin", branch="main", enabled=True,
                        retry_schedule=(3600.0,))
    worker.stop(flush_seconds=5.0)
    return Runtime(
        audit=audit, push_worker=worker,
        refresher=IndexRefresher(root, audit=audit, enabled=False),
    )


def _commit_note(repo: Path, rel: str, text: str, message: str,
                 run_git: GitRunner) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message)


def _hand_edit(obsidian: Path, rel: str, text: str, message: str,
               run_git: GitRunner) -> None:
    """The owner edits a note in their own clone and pushes it."""
    _commit_note(obsidian, rel, text, message, run_git)
    run_git(obsidian, "push", "-q", "origin", "main")


def _audit_rows(root: Path) -> list[dict[str, str | None]]:
    path = root / "logs" / "mcp-audit.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, str | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed: dict[str, str | None] = json.loads(line)
            rows.append(parsed)
    return rows


def _outcomes(root: Path) -> list[str | None]:
    return [row["outcome"] for row in _audit_rows(root)]


@pytest.fixture()
def as_agent_a() -> Iterator[None]:
    token = AGENT_VAR.set("agent-a")
    try:
        yield
    finally:
        AGENT_VAR.reset(token)


def _cfg(make_cfg: CfgFactory, root: Path) -> ServerConfig:
    return make_cfg(root)


# ------------------------------------------------------- path intersection

def test_touches_matches_the_note_being_written() -> None:
    assert sync.touches(("knowledge/notes/shared.md",), "knowledge/notes/shared.md")


def test_touches_ignores_an_unrelated_note() -> None:
    assert not sync.touches(("knowledge/notes/other.md",), "knowledge/notes/new.md")


def test_touches_matches_a_slug_against_its_note() -> None:
    """The concept tool is called with a slug, the inbox tool with an
    inbox-relative path; git reports full vault paths."""
    assert sync.touches(("knowledge/concepts/alpha.md",), "alpha")
    assert sync.touches(("inbox/drop/report.pdf",), "drop/report.pdf")
    assert sync.touches(("knowledge/people/anna.md",), "people/anna")


def test_touches_treats_an_unknown_target_as_a_match() -> None:
    """Unable to prove the write is unrelated, refuse rather than write."""
    assert sync.touches(("knowledge/notes/other.md",), None)


# ------------------------------------------------------------- HEAD guard

def test_refuse_if_head_moved_is_a_noop_outside_a_write_section(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    _commit_note(mcp_vault, "note.md", "seed\n", "seed", run_git)
    sync.refuse_if_head_moved()   # no section open: nothing to compare


def test_refuse_if_head_moved_fires_when_another_commit_lands(
    mcp_vault: Path, run_git: GitRunner
) -> None:
    _commit_note(mcp_vault, "note.md", "seed\n", "seed", run_git)
    with sync.write_section(mcp_vault):
        sync.refuse_if_head_moved()   # still the commit we started on
        _commit_note(mcp_vault, "note.md", "moved\n", "moved", run_git)
        with pytest.raises(ToolError, match="HEAD moved"):
            sync.refuse_if_head_moved()
    sync.refuse_if_head_moved()   # section closed again


# ---------------------------------------------- a conflict on another note

def test_a_conflict_on_an_unrelated_note_does_not_refuse_the_write(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory, as_agent_a: None
) -> None:
    """Review 2026-09-09 #3: a rebase replays EVERY unpushed local commit,
    so a hand edit conflicting on note X used to refuse — permanently —
    every write to unrelated note Y."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    runtime = _pushing_runtime(server)
    cfg = _cfg(make_cfg, server)
    # An unpushed server commit and a conflicting hand edit, both on other.md.
    _commit_note(server, "knowledge/notes/other.md", "agent version\n",
                 "agent edits other", run_git)
    _hand_edit(obsidian, "knowledge/notes/other.md", "hand version\n",
               "hand edits other", run_git)

    result = tools_mod.tool_create_note(
        cfg, runtime, path="knowledge/notes/brand-new.md", content="# unrelated\n"
    )

    assert result.committed is True
    assert (server / "knowledge" / "notes" / "brand-new.md").exists()
    # And it stays unwedged: a second write is not refused either.
    second = tools_mod.tool_create_note(
        cfg, runtime, path="knowledge/notes/brand-new-2.md", content="# also\n"
    )
    assert second.committed is True
    # The standing conflict is reported, not swallowed — in the audit log
    # and in the worker's status, naming the only recovery there is.
    conflict_rows = [r for r in _audit_rows(server) if r["outcome"] == "pull-conflict"]
    assert len(conflict_rows) == 2
    detail = conflict_rows[0]["detail"]
    assert detail is not None
    assert "knowledge/notes/other.md" in detail
    assert "git rebase" in detail
    sync_error = runtime.push_worker.status()["sync_error"]
    assert sync_error is not None and "knowledge/notes/other.md" in sync_error


def test_a_conflict_on_the_written_note_refuses_and_names_the_recovery(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory, as_agent_a: None
) -> None:
    """IMP-029 itself: the one case a refusal is right. The message must
    not promise a retry — no retry settles a conflict."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    runtime = _pushing_runtime(server)
    cfg = _cfg(make_cfg, server)
    _commit_note(server, "knowledge/notes/shared.md", "agent version\n",
                 "agent edits shared", run_git)
    _hand_edit(obsidian, "knowledge/notes/shared.md", "hand version\n",
               "hand edits shared", run_git)
    head_before = run_git(server, "rev-parse", "HEAD")

    with pytest.raises(ToolError, match="settle it by hand") as raised:
        tools_mod.tool_append_to_note(
            cfg, runtime, path="knowledge/notes/shared.md", content="more\n"
        )

    assert "re-read the note and retry" not in str(raised.value)
    assert run_git(server, "rev-parse", "HEAD") == head_before
    assert (server / "knowledge" / "notes" / "shared.md").read_text(
        encoding="utf-8"
    ) == "agent version\n"
    assert any(str(o).startswith("refused:") for o in _outcomes(server))


# ------------------------------------------------------------ blocked pull

def test_a_blocked_pull_is_audited_and_the_write_still_lands(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory, as_agent_a: None
) -> None:
    """Review 2026-09-09 #5: an untracked file where an incoming commit
    adds one makes the fast-forward fail. That used to escape as a bare
    GitError, be logged, and leave a clean ``ok`` audit row while the clone
    silently stopped syncing — the exact failure AUD-122 exists to end."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    runtime = _pushing_runtime(server)
    cfg = _cfg(make_cfg, server)
    (server / "knowledge" / "notes" / "collide.md").write_text(
        "server scratch\n", encoding="utf-8"
    )
    _hand_edit(obsidian, "knowledge/notes/collide.md", "hand version\n",
               "hand adds collide", run_git)

    result = tools_mod.tool_create_note(
        cfg, runtime, path="knowledge/notes/agent.md", content="# a\n"
    )

    assert result.committed is True
    failed = [r for r in _audit_rows(server) if r["outcome"] == "pull-failed"]
    assert len(failed) == 1
    detail = failed[0]["detail"]
    assert detail is not None and "collide.md" in detail
    assert runtime.push_worker.status()["sync_error"] is not None
    # Untouched: the blocker is the human's to clear, not the server's.
    assert (server / "knowledge" / "notes" / "collide.md").read_text(
        encoding="utf-8"
    ) == "server scratch\n"


# ------------------------------------------------------- interrupted rebase

def test_an_interrupted_rebase_refuses_with_the_remedy_named(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory, as_agent_a: None
) -> None:
    """Review 2026-09-09 #6: a killed rebase leaves a CLEAN tree on a
    detached HEAD, so the branch message alone sent the reader looking for
    a stray checkout instead of the rebase that caused it."""
    server, obsidian = _clone_pair(tmp_path, run_git)
    runtime = _pushing_runtime(server)
    cfg = _cfg(make_cfg, server)
    for i in (1, 2):
        _commit_note(server, f"knowledge/notes/l{i}.md", f"l{i}\n", f"local {i}", run_git)
    _hand_edit(obsidian, "knowledge/notes/hand.md", "hand\n", "hand edit", run_git)
    run_git(server, "fetch", "-q", "origin", "main")
    # --exec false stops the rebase after its first replay: a kill between
    # steps, leaving .git/rebase-merge behind.
    subprocess.run(["git", "-C", str(server), "rebase", "--exec", "false", "FETCH_HEAD"],
                   capture_output=True, text=True, check=False)
    assert (server / ".git" / "rebase-merge").exists()
    assert run_git(server, "status", "--porcelain", "--untracked-files=no") == ""

    with pytest.raises(ToolError, match="git rebase --abort"):
        tools_mod.tool_create_note(
            cfg, runtime, path="knowledge/notes/next.md", content="# n\n"
        )

    assert not (server / "knowledge" / "notes" / "next.md").exists()
    assert any(str(o).startswith("refused:") for o in _outcomes(server))


# ------------------------------------------- HEAD moving under a tool body

def test_a_rebase_between_read_and_write_refuses_instead_of_overwriting(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory, as_agent_a: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2026-09-09 #2: the IMP-029 loss with no conflict in sight.

    The tool reads the note; a rebase then absorbs a hand edit to it — in
    production the push worker's retry, which fires precisely when a hand
    edit has just landed, or any other process sharing the clone. The tool
    then wrote its stale transform over the absorbed edit and audited
    ``ok``. The hook below stands in for that rebase, timed inside the tool
    body the same way.
    """
    server, obsidian = _clone_pair(tmp_path, run_git)
    runtime = _pushing_runtime(server)
    cfg = _cfg(make_cfg, server)
    rel = "knowledge/notes/shared.md"
    fired: list[bool] = []

    def hooked(
        content: str,
        *,
        agent: str,
        mode: Literal["create", "replace", "append"],
        memory_area: bool,
        prior: str | None = None,
    ) -> str:
        if not fired:
            fired.append(True)
            _hand_edit(obsidian, rel, "---\ntitle: shared\n---\n\nHAND EDIT, precious\n",
                       "hand edit", run_git)
            pull_rebase(server, remote="origin", branch="main", expected_branch="main")
        return real_stamp(
            content, agent=agent, mode=mode, memory_area=memory_area, prior=prior
        )

    monkeypatch.setattr(tools_mod, "stamp_provenance", hooked)

    with pytest.raises(ToolError, match="HEAD moved"):
        tools_mod.tool_append_to_note(
            cfg, runtime, path=rel, content="appended by agent\n"
        )

    assert fired == [True]
    assert "HAND EDIT, precious" in (server / rel).read_text(encoding="utf-8")
    assert "appended by agent" not in (server / rel).read_text(encoding="utf-8")
    assert any(str(o).startswith("refused:") for o in _outcomes(server))


# --------------------------------------------------------------- the lock

def test_concurrent_writes_to_distinct_notes_all_commit(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2026-09-09 #1: the pull ran outside the write lock, so one
    thread's pull inspected the tree another thread had written to but not
    yet committed, and refused the write over a dirty path no agent could
    commit or discard. Nothing here is contrived — the delay only widens
    the window that every real write has between its disk write and its
    commit."""
    server, _obsidian = _clone_pair(tmp_path, run_git)
    runtime = _pushing_runtime(server)
    cfg = _cfg(make_cfg, server)
    for i in range(4):
        _commit_note(server, f"knowledge/notes/t{i}.md", f"---\ntitle: t{i}\n---\n\nbody\n",
                     f"seed t{i}", run_git)
    run_git(server, "push", "-q", "origin", "main")

    real_commit = tools_mod._commit

    def slow_commit(cfg_: ServerConfig, paths: list[Path], message: str) -> CommitOutcome:
        # The window between "written to disk" and "committed", widened.
        threading.Event().wait(0.05)
        return real_commit(cfg_, paths, message)

    monkeypatch.setattr(tools_mod, "_commit", slow_commit)

    results: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        token = AGENT_VAR.set(f"agent-{index}")
        try:
            for k in range(3):
                try:
                    tools_mod.tool_append_to_note(
                        cfg, runtime, path=f"knowledge/notes/t{index}.md",
                        content=f"x{k}\n",
                    )
                    outcome = "ok"
                except Exception as exc:   # noqa: BLE001 — the point is what fails
                    outcome = f"{type(exc).__name__}: {exc}"
                with lock:
                    results.append(outcome)
        finally:
            AGENT_VAR.reset(token)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)

    assert results == ["ok"] * 12, [r for r in results if r != "ok"][:3]
    assert run_git(server, "status", "--porcelain", "--untracked-files=no") == ""
    for i in range(4):
        body = (server / "knowledge" / "notes" / f"t{i}.md").read_text(encoding="utf-8")
        assert ["x0", "x1", "x2"] == [ln for ln in body.splitlines() if ln.startswith("x")]
