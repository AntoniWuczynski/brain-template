"""Commit and push helpers for write tools.

All git invocations use explicit argv (no shell). Output goes to the
log, not back to the client, so we never echo a remote URL or commit
hash into the MCP response.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Serialise all git operations: the working tree + index is shared
# global state and tool calls run concurrently in a threadpool. Two
# unsynchronised commits race the index (index.lock) and can stage the
# wrong file set.
_GIT_LOCK = threading.Lock()


class GitError(Exception):
    pass


@dataclass
class CommitOutcome:
    """Outcome of a write's git step. ``detail`` is safe to show the agent
    (no remote URLs / ssh hints — those stay in the server log)."""
    sha: str | None
    committed: bool
    pushed: bool
    detail: str | None = None


def _git(cwd: Path, *args: str, timeout: float = 30.0) -> str:
    """Run a git command with explicit argv. Returns stdout, raises GitError
    on failure or timeout. A timeout matters most for ``push``: without it a
    black-holed network route would hang forever holding the git lock and
    wedging every write tool."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise GitError(f"git {args[0] if args else '?'} timed out after {timeout}s") from None
    if result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} exited {result.returncode}: {result.stderr.strip()[:300]}"
        )
    return result.stdout


def commit_paths(
    vault_root: Path,
    *,
    paths: list[Path],
    message: str,
) -> CommitOutcome:
    """Stage the given paths and commit — no push.

    The whole add/status/commit/rev-parse sequence holds ``_GIT_LOCK``
    because the index is shared global state across concurrent tool
    threads. Returns ``pushed=False`` always; pushing is the caller's
    business (synchronously via ``commit_and_push`` or asynchronously
    via ``push_queue.PushWorker``). Raises ``GitError`` only if the
    commit step itself fails.
    """
    if not paths:
        return CommitOutcome(None, False, False, "no paths given")

    with _GIT_LOCK:
        # Stage only the explicit paths we touched.
        rel_paths = []
        for p in paths:
            try:
                rel_paths.append(str(p.resolve().relative_to(vault_root)))
            except ValueError:
                raise GitError(f"path is not inside the vault: {p}") from None

        _git(vault_root, "add", "--", *rel_paths)

        # No-op (e.g. appending identical content): git commit returns 1.
        status = _git(vault_root, "status", "--porcelain", "--", *rel_paths).strip()
        if not status:
            log.info("no changes to commit for: %s", rel_paths)
            return CommitOutcome(None, False, False, "no changes to commit")

        _git(vault_root, "commit", "-m", message, "--", *rel_paths)
        sha = _git(vault_root, "rev-parse", "HEAD").strip()
        return CommitOutcome(sha, True, False, None)


def commit_and_push(
    vault_root: Path,
    *,
    paths: list[Path],
    message: str,
    remote: str,
    branch: str,
    push: bool = True,
) -> CommitOutcome:
    """Stage the given paths, commit, optionally push (synchronously).

    Returns a CommitOutcome describing exactly what happened so the caller
    can tell the agent the truth (committed? pushed?) instead of assuming
    success. Raises ``GitError`` only if the commit step itself fails.

    The push runs OUTSIDE ``_GIT_LOCK``: push only reads refs, never the
    index or working tree, so a concurrent commit can't be corrupted —
    worst case the push misses it and the next push catches up.
    """
    outcome = commit_paths(vault_root, paths=paths, message=message)
    if not outcome.committed:
        return outcome
    sha = outcome.sha

    if not push:
        return CommitOutcome(sha, True, False, "push disabled")

    try:
        _git(vault_root, "push", remote, branch, timeout=15.0)
    except GitError as exc:
        # Commit is on disk; surface a generic warning to the agent and
        # keep the real git stderr (remote URL / ssh hints) in the log.
        log.warning("commit %s landed locally but push to %s/%s failed: %s",
                    (sha or "?")[:8], remote, branch, exc)
        return CommitOutcome(sha, True, False, "committed locally but push to remote failed")

    return CommitOutcome(sha, True, True, None)
