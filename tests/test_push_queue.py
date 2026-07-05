"""Tests for mcp_server.push_queue — the async coalescing push worker.

Uses a throwaway local git repo pushing to a bare repo on the same
filesystem (file transport), so the pushes are REAL but fully offline
and fast. Timeouts are kept tight; the whole module adds a few seconds.
"""
from __future__ import annotations

import subprocess
import sys
import time
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


def _wait_for(cond, timeout: float = 8.0, interval: float = 0.02) -> bool:
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
    assert worker.status() == {"state": "idle", "consecutive_failures": 0, "last_error": None}


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
