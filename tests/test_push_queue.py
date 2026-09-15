"""Tests for mcp_server.push_queue — the async coalescing push worker.

Uses a throwaway local git repo pushing to a bare repo on the same
filesystem (file transport), so the pushes are REAL but fully offline
and fast. Timeouts are kept tight; the whole module adds a few seconds.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

# mcp_server is not an installed package (only ingest_lib is). The full
# suite imports it via a collection-order side effect; pin the repo root
# onto sys.path so this file also runs standalone.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp_server.push_queue import PushWorker  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _make_repos(tmp_path: Path) -> tuple[Path, Path]:
    """A working repo with one commit + a bare repo wired as its origin."""
    work = tmp_path / "work"
    bare = tmp_path / "bare.git"
    work.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "test")
    (work / "seed.md").write_text("seed\n", encoding="utf-8")
    _git(work, "add", "seed.md")
    _git(work, "commit", "-q", "-m", "seed")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    _git(work, "remote", "add", "origin", str(bare))
    return work, bare


def _commit(work: Path, name: str) -> str:
    (work / name).write_text(f"{name}\n", encoding="utf-8")
    _git(work, "add", name)
    _git(work, "commit", "-q", "-m", name)
    return _git(work, "rev-parse", "HEAD")


def _bare_head(bare: Path) -> str | None:
    try:
        return _git(bare, "rev-parse", "main")
    except subprocess.CalledProcessError:
        return None  # branch not pushed yet


def _wait_for(cond: Callable[[], bool], timeout: float = 8.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def test_enabled_push_lands_on_remote(tmp_path: Path) -> None:
    work, bare = _make_repos(tmp_path)
    worker = PushWorker(work, remote="origin", branch="main", enabled=True)
    try:
        sha = _git(work, "rev-parse", "HEAD")
        assert worker.request_push() == "queued"
        assert _wait_for(lambda: _bare_head(bare) == sha), "push never reached the bare remote"
        assert _wait_for(lambda: worker.status()["state"] == "idle")
        status = worker.status()
        assert status["consecutive_failures"] == 0
        assert status["last_error"] is None
    finally:
        worker.stop(flush_seconds=1.0)


def test_rapid_requests_coalesce_to_final_sha(tmp_path: Path) -> None:
    work, bare = _make_repos(tmp_path)
    worker = PushWorker(work, remote="origin", branch="main", enabled=True)
    try:
        final = ""
        for i in range(5):
            final = _commit(work, f"note-{i}.md")
            assert worker.request_push() == "queued"
        # One push carries the whole branch, so however the requests
        # coalesced, the remote must converge on the LAST commit.
        assert _wait_for(lambda: _bare_head(bare) == final), "remote never converged"
        assert _wait_for(lambda: worker.status()["state"] == "idle")
    finally:
        worker.stop(flush_seconds=1.0)


def test_disabled_returns_disabled_and_starts_no_thread(tmp_path: Path) -> None:
    work, _bare = _make_repos(tmp_path)
    worker = PushWorker(work, remote="origin", branch="main", enabled=False)
    assert worker.request_push() == "disabled"
    assert worker._thread is None  # lazy start never happened
    assert worker.status() == {
        "state": "idle", "consecutive_failures": 0,
        "last_error": None, "sync_error": None,
    }


def test_failure_retries_then_recovers_after_remote_fixed(tmp_path: Path) -> None:
    work, bare = _make_repos(tmp_path)
    missing = tmp_path / "missing.git"  # nonexistent: every push fails fast
    _git(work, "remote", "set-url", "origin", str(missing))
    # Tiny schedule so the failure->retry->recovery cycle fits in well
    # under a second of waiting.
    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(0.05, 0.1),
    )
    try:
        sha = _git(work, "rev-parse", "HEAD")
        assert worker.request_push() == "queued"
        assert _wait_for(
            lambda: worker.status()["state"] == "retrying"
            and worker.status()["consecutive_failures"] >= 1
        ), "worker never entered retrying"
        status = worker.status()
        # Sanitized: the failing remote path must not leak via status().
        assert status["last_error"] is not None
        assert str(missing) not in status["last_error"]

        # Fix the remote; the backoff loop must pick it up and recover.
        _git(work, "remote", "set-url", "origin", str(bare))
        assert _wait_for(lambda: _bare_head(bare) == sha), "push never recovered"
        assert _wait_for(lambda: worker.status()["state"] == "idle")
        status = worker.status()
        assert status["consecutive_failures"] == 0
        assert status["last_error"] is None
    finally:
        worker.stop(flush_seconds=1.0)


def test_startup_push_flushes_a_commit_stranded_by_a_crash(tmp_path: Path) -> None:
    """AUD-040: a commit left unpushed by a crash must reach the remote when
    the worker is next constructed, without waiting for another write."""
    work, bare = _make_repos(tmp_path)
    stranded = _commit(work, "stranded.md")
    worker = PushWorker(work, remote="origin", branch="main", enabled=True)
    try:
        # No request_push() call: construction alone must flush the branch.
        assert _wait_for(lambda: _bare_head(bare) == stranded), \
            "startup never pushed the stranded commit"
        assert _wait_for(lambda: worker.status()["state"] == "idle")
    finally:
        worker.stop(flush_seconds=1.0)


# AUD-099: a vault that has silently stopped being backed up must show up
# somewhere a human or an agent looks. status() is not exposed over MCP, so
# the audit log is the surface.

def _audit_rows(work: Path) -> list[dict[str, object]]:
    path = work / "logs" / "mcp-audit.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed: dict[str, object] = json.loads(line)
            rows.append(parsed)
    return rows


def _push_outcomes(work: Path) -> list[object]:
    return [r["outcome"] for r in _audit_rows(work) if r["tool"] == "push"]


def test_repeated_push_failures_are_audited_and_the_alarm_clears(tmp_path: Path) -> None:
    work, bare = _make_repos(tmp_path)
    missing = tmp_path / "missing.git"
    _git(work, "remote", "set-url", "origin", str(missing))
    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(0.02,),
    )
    try:
        sha = _git(work, "rev-parse", "HEAD")
        assert _wait_for(lambda: "failing" in _push_outcomes(work)), \
            "a run of failed pushes was never audited"
        alarm = [r for r in _audit_rows(work) if r["outcome"] == "failing"][0]
        assert alarm["agent"] == "system"
        detail = alarm["detail"]
        assert isinstance(detail, str) and str(missing) not in detail

        # Only one alarm per run of failures, not one per retry.
        assert _push_outcomes(work).count("failing") == 1

        _git(work, "remote", "set-url", "origin", str(bare))
        assert _wait_for(lambda: _bare_head(bare) == sha), "push never recovered"
        assert _wait_for(lambda: "recovered" in _push_outcomes(work)), \
            "recovery was never audited"
    finally:
        worker.stop(flush_seconds=1.0)


def test_final_flush_failure_at_shutdown_is_audited(tmp_path: Path) -> None:
    work, _bare = _make_repos(tmp_path)
    missing = tmp_path / "missing.git"
    _git(work, "remote", "set-url", "origin", str(missing))
    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(5.0,),
    )
    assert _wait_for(lambda: worker.status()["state"] == "retrying")
    worker.stop(flush_seconds=0.5)
    assert "unpushed" in _push_outcomes(work), \
        "commits stranded by shutdown were not audited"


# AUD-122: the vault has two writers now (this server and the owner's
# Obsidian clone), so a push can be rejected for being behind. The worker
# must absorb the other side and retry once — and never resolve a conflict.

def _clone_of(tmp_path: Path, bare: Path, name: str) -> Path:
    """A second working clone of the same bare repo — the Obsidian side."""
    clone = tmp_path / name
    subprocess.run(["git", "clone", "-q", str(bare), str(clone)], check=True)
    _git(clone, "config", "user.email", "hand@example.com")
    _git(clone, "config", "user.name", "hand")
    return clone


def _subjects(repo: Path, *args: str) -> list[str]:
    out = _git(repo, "log", "--format=%s", *args)
    return out.splitlines() if out else []


def test_push_rejected_as_non_fast_forward_rebases_and_retries(tmp_path: Path) -> None:
    work, bare = _make_repos(tmp_path)
    _git(work, "push", "-q", "origin", "main")
    obsidian = _clone_of(tmp_path, bare, "obsidian")
    _commit(obsidian, "hand.md")
    _git(obsidian, "push", "-q", "origin", "main")
    _commit(work, "agent.md")

    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(0.05,),
    )
    try:
        # The first push is rejected; the retry after the rebase is what
        # puts the agent's commit on the remote, on top of the hand edit.
        assert _wait_for(lambda: _subjects(bare, "main")[:2] == ["agent.md", "hand.md"]), \
            "the rejected push never recovered by rebasing"
        assert _subjects(work)[:2] == ["agent.md", "hand.md"]
        assert _git(work, "status", "--porcelain") == ""
        assert _wait_for(lambda: worker.status()["state"] == "idle")
        assert worker.status()["consecutive_failures"] == 0
    finally:
        worker.stop(flush_seconds=1.0)


def test_push_conflict_is_surfaced_and_never_resolved(tmp_path: Path) -> None:
    work, bare = _make_repos(tmp_path)
    _git(work, "push", "-q", "origin", "main")
    obsidian = _clone_of(tmp_path, bare, "obsidian")
    (obsidian / "seed.md").write_text("edited by hand\n", encoding="utf-8")
    _git(obsidian, "add", "seed.md")
    _git(obsidian, "commit", "-q", "-m", "hand edit")
    _git(obsidian, "push", "-q", "origin", "main")
    remote_head = _bare_head(bare)

    (work / "seed.md").write_text("edited by the agent\n", encoding="utf-8")
    _git(work, "add", "seed.md")
    _git(work, "commit", "-q", "-m", "agent edit")
    local_head = _git(work, "rev-parse", "HEAD")

    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(5.0,),
    )
    try:
        assert _wait_for(lambda: worker.status()["last_error"] is not None), \
            "the conflicting push never surfaced an error"
        error = worker.status()["last_error"]
        assert error is not None and "seed.md" in error and "conflict" in error
        # Neither side won: local branch, working tree and remote unchanged.
        assert _git(work, "rev-parse", "HEAD") == local_head
        assert _git(work, "status", "--porcelain") == ""
        assert (work / "seed.md").read_text(encoding="utf-8") == "edited by the agent\n"
        assert _bare_head(bare) == remote_head
    finally:
        worker.stop(flush_seconds=1.0)


def test_push_conflict_also_names_the_manual_recovery_in_sync_error(
    tmp_path: Path
) -> None:
    """Review 2026-09-09 #3: a conflict is not a retryable push failure.
    ``last_error`` reads as one, so the condition a human has to settle
    gets its own field, with the recovery named."""
    work, bare = _make_repos(tmp_path)
    _git(work, "push", "-q", "origin", "main")
    obsidian = _clone_of(tmp_path, bare, "obsidian")
    (obsidian / "seed.md").write_text("edited by hand\n", encoding="utf-8")
    _git(obsidian, "add", "seed.md")
    _git(obsidian, "commit", "-q", "-m", "hand edit")
    _git(obsidian, "push", "-q", "origin", "main")
    (work / "seed.md").write_text("edited by the agent\n", encoding="utf-8")
    _git(work, "add", "seed.md")
    _git(work, "commit", "-q", "-m", "agent edit")

    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(5.0,),
    )
    try:
        assert _wait_for(lambda: worker.status()["sync_error"] is not None), \
            "the standing conflict was never surfaced in status()"
        sync_error = worker.status()["sync_error"]
        assert sync_error is not None
        assert "seed.md" in sync_error and "git rebase" in sync_error
    finally:
        worker.stop(flush_seconds=1.0)


def test_the_worker_never_rebases_a_branch_it_was_not_configured_for(
    tmp_path: Path
) -> None:
    """Review 2026-09-09 #4: parked on another branch (a human debugging),
    the worker rebased THAT branch onto the remote's main — rewriting
    unrelated history, and leaving main, the branch it meant to fix,
    untouched so the retry push failed anyway."""
    work, bare = _make_repos(tmp_path)
    _git(work, "push", "-q", "origin", "main")
    obsidian = _clone_of(tmp_path, bare, "obsidian")
    _commit(obsidian, "hand.md")
    _git(obsidian, "push", "-q", "origin", "main")
    _commit(work, "local.md")           # unpushed commit on main
    main_head = _git(work, "rev-parse", "main")
    _git(work, "checkout", "-q", "-b", "sidebar")
    side_head = _commit(work, "side.md")

    worker = PushWorker(
        work, remote="origin", branch="main", enabled=True,
        retry_schedule=(5.0,),
    )
    try:
        assert _wait_for(lambda: worker.status()["sync_error"] is not None), \
            "the refused rebase was never surfaced"
        sync_error = worker.status()["sync_error"]
        assert sync_error is not None and "sidebar" in sync_error
        assert _git(work, "rev-parse", "sidebar") == side_head
        assert _git(work, "rev-parse", "main") == main_head
        assert _subjects(work) == ["side.md", "local.md", "seed"]
    finally:
        worker.stop(flush_seconds=1.0)
