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


def current_branch(vault_root: Path) -> str:
    """Short name of the checked-out branch, or ``'(detached HEAD)'``."""
    try:
        return _git(vault_root, "symbolic-ref", "--short", "HEAD").strip()
    except GitError:
        return "(detached HEAD)"


def head_sha(vault_root: Path) -> str | None:
    """The commit HEAD points at, or None when git can't say (a repo with
    no commits yet). Cheap enough — one ``rev-parse`` — to re-read on the
    write path as a staleness check."""
    try:
        return _git(vault_root, "rev-parse", "HEAD").strip()
    except GitError:
        return None


def rebase_in_progress(vault_root: Path) -> bool:
    """Whether a rebase is stopped part-way through in this clone.

    git leaves ``rebase-merge`` (interactive/merge backend) or
    ``rebase-apply`` (am backend) in the git dir for the whole run and
    removes it on finish or ``--abort``. Either present means HEAD is
    parked mid-replay: usually detached, and the working tree may be
    clean, so neither the dirty check nor a branch compare explains it.
    ``rev-parse --git-path`` resolves the git dir for us, so this holds
    for a worktree or a ``.git`` file as well as a plain checkout.
    """
    for name in ("rebase-merge", "rebase-apply"):
        try:
            raw = _git(vault_root, "rev-parse", "--git-path", name).strip()
        except GitError:
            return False
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = vault_root / candidate
        if candidate.exists():
            return True
    return False


def commit_paths(
    vault_root: Path,
    *,
    paths: list[Path],
    message: str,
    expected_branch: str | None = None,
) -> CommitOutcome:
    """Stage the given paths and commit — no push.

    The whole add/status/commit/rev-parse sequence holds ``_GIT_LOCK``
    because the index is shared global state across concurrent tool
    threads. Returns ``pushed=False`` always; pushing is the caller's
    business (asynchronously via ``push_queue.PushWorker``). Raises
    ``GitError`` only if the commit step itself fails, or, when
    ``expected_branch`` is given, if a different branch (or no branch,
    i.e. detached HEAD) is checked out.
    """
    if not paths:
        return CommitOutcome(None, False, False, "no paths given")

    with _GIT_LOCK:
        if expected_branch is not None:
            current = current_branch(vault_root)
            if current != expected_branch:
                raise GitError(
                    f"refusing to commit: checked-out branch {current!r} is not "
                    f"the configured {expected_branch!r}"
                )

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


# ------------------------------------------------------- pull before write

# Upper bound on a single `git fetch`, for the same reason `push` has one:
# after the migration both this clone and the owner's Obsidian clone write
# to one remote, so every write fetches — a black-holed route must fail
# fast rather than hold the git lock and wedge the write path.
_FETCH_TIMEOUT_S = 15.0


class DirtyTreeError(GitError):
    """A rebase was attempted with uncommitted changes to tracked files.

    ``paths`` names them. Every writer of tracked files inside this process
    — the write tools and the background index refresher — holds
    ``sync.WRITE_LOCK`` from its first byte on disk through its commit, and
    this check runs under that same lock, so a dirty tree seen here is work
    from OUTSIDE the server (a hand edit in the shared clone, an interrupted
    script), never the server's own half-finished write.
    """

    def __init__(self, message: str, paths: tuple[str, ...]) -> None:
        super().__init__(message)
        self.paths = paths


class RebaseConflictError(GitError):
    """The remote's commits conflict with this clone's on ``paths``.

    Raised only after the rebase has been aborted, so the branch and the
    working tree are exactly where they were. Never resolved automatically
    in either direction (FOUNDER_DECISIONS IMP-029).
    """

    def __init__(self, message: str, paths: tuple[str, ...]) -> None:
        super().__init__(message)
        self.paths = paths


@dataclass
class RebaseOutcome:
    """What one ``pull_rebase`` did.

    ``fetched`` is False when the remote was unreachable — the caller may
    carry on writing locally. ``absorbed`` counts the commits taken from
    the remote; ``rebased`` distinguishes a real rebase (this clone had
    commits of its own to replay) from a fast-forward.

    ``blocked`` is set when the remote WAS reached but its commits could
    not be applied for a reason that is neither a conflict nor an
    unreachable remote — an untracked file standing where an incoming
    commit adds one, a broken repo state. It carries a short cause. The
    caller is expected to record it and carry on writing locally: the
    clone is behind and will stay behind until a human clears the blocker,
    which is a condition to REPORT, not one to fail an agent's write over.
    """

    fetched: bool
    absorbed: int
    rebased: bool
    detail: str | None = None
    blocked: str | None = None


def _dirty_tracked_paths(vault_root: Path) -> tuple[str, ...]:
    """Tracked paths with staged or unstaged changes. Untracked files are
    deliberately ignored: a real vault always has some (scratch notes, an
    unprocessed inbox) and they neither block a rebase nor mean the server
    left a write half-done."""
    out = _git(vault_root, "status", "--porcelain", "--untracked-files=no")
    return tuple(line[3:] for line in out.splitlines() if line.strip())


def _unmerged_paths(vault_root: Path) -> tuple[str, ...]:
    """Paths git left in conflict, read before the rebase is aborted."""
    try:
        out = _git(vault_root, "diff", "--name-only", "--diff-filter=U")
    except GitError:
        return ()
    return tuple(line for line in out.splitlines() if line.strip())


def _count_commits(vault_root: Path, range_spec: str) -> int:
    return int(_git(vault_root, "rev-list", "--count", range_spec).strip())


def _blocked_outcome(exc: GitError) -> RebaseOutcome:
    """The remote answered but its commits could not be applied.

    The cause is git's complaint flattened to one line and capped. Only the
    ``fetch`` step can carry a remote URL or ssh hint in its stderr, and a
    failed fetch never reaches here (it returns ``fetched=False``), so this
    string is safe to put in the audit log and in ``PushWorker.status()``.
    """
    cause = " ".join(part.strip() for part in str(exc).splitlines() if part.strip())
    return RebaseOutcome(
        fetched=True, absorbed=0, rebased=False, blocked=cause[:200]
    )


def pull_rebase(
    vault_root: Path,
    *,
    remote: str,
    branch: str,
    expected_branch: str | None = None,
    timeout: float = _FETCH_TIMEOUT_S,
) -> RebaseOutcome:
    """Absorb the remote's commits into the checked-out branch (AUD-122).

    Fast-forwards when this clone has no commits the remote lacks, and
    rebases the local-only commits on top of the remote otherwise. Holds
    ``_GIT_LOCK`` throughout: it moves HEAD and the working tree, so it
    must not interleave with a commit. Callers that also write files —
    the write path and the push worker — hold ``sync.WRITE_LOCK`` across
    this call and their own commit, so nothing can move HEAD between a
    tool's read of a note and the commit of its rewrite.

    ``expected_branch`` is the guard ``commit_paths`` already has: this
    moves HEAD and rewrites history, so on anything but the configured
    branch (a human's debugging checkout, a detached HEAD, an interrupted
    rebase) it raises ``GitError`` and touches nothing. Both production
    callers pass it; it stays optional so a caller that has already
    checked can say so by omission.

    Four outcomes, deliberately distinct:

    - **Unreachable remote** (the laptop off the network) is NOT an error.
      It returns ``fetched=False`` with a ``detail``; the caller writes and
      commits locally and the next push carries both sides.
    - **Blocked** — the remote was reached but its commits would not
      apply, and not because of a conflict: an untracked file standing
      where an incoming commit adds one is the common case. Returns
      ``blocked=<cause>``; the branch has not moved. The caller records it
      (the clone is behind) and carries on locally.
    - **Dirty tree** raises ``DirtyTreeError``. Every in-process writer
      holds ``sync.WRITE_LOCK`` from disk to commit and this runs under
      that lock, so a dirty tree means a foreign edit is in flight and a
      rebase would carry it — refuse instead.
    - **Conflict** raises ``RebaseConflictError`` after ``git rebase
      --abort``, leaving the branch exactly where it was. Note the conflict
      is over the WHOLE stack of unpushed local commits, so ``paths`` may
      name notes this caller's write never touches — the caller decides
      what that means for it (``sync.pull_before_write`` refuses only when
      the paths intersect its own).
    """
    with _GIT_LOCK:
        if expected_branch is not None:
            current = current_branch(vault_root)
            if current != expected_branch:
                raise GitError(
                    f"refusing to rebase: checked-out branch {current!r} is not "
                    f"the configured {expected_branch!r}"
                )

        dirty = _dirty_tracked_paths(vault_root)
        if dirty:
            raise DirtyTreeError(
                "working tree has uncommitted changes to: " + ", ".join(dirty),
                dirty,
            )

        try:
            _git(vault_root, "fetch", remote, branch, timeout=timeout)
        except GitError as exc:
            # Offline is the normal case on a laptop; log and let the write
            # proceed locally rather than failing it.
            log.warning("pull-before-write: fetch from %s/%s failed: %s", remote, branch, exc)
            return RebaseOutcome(
                fetched=False, absorbed=0, rebased=False,
                detail="fetch failed (details in server log)",
            )

        try:
            absorbed = _count_commits(vault_root, "HEAD..FETCH_HEAD")
            if absorbed == 0:
                return RebaseOutcome(fetched=True, absorbed=0, rebased=False)
            local_only = _count_commits(vault_root, "FETCH_HEAD..HEAD")
        except GitError as exc:
            log.warning("pull-before-write: reading the divergence against %s/%s "
                        "failed: %s", remote, branch, exc)
            return _blocked_outcome(exc)

        if local_only == 0:
            try:
                _git(vault_root, "merge", "--ff-only", "FETCH_HEAD")
            except GitError as exc:
                # e.g. an untracked file at a path the incoming commit adds.
                # Nothing moved; report it rather than raising into a write.
                log.warning("pull-before-write: fast-forward from %s/%s blocked: %s",
                            remote, branch, exc)
                return _blocked_outcome(exc)
            log.info("pull-before-write: fast-forwarded %d commit(s) from %s/%s",
                     absorbed, remote, branch)
            return RebaseOutcome(fetched=True, absorbed=absorbed, rebased=False)

        try:
            _git(vault_root, "rebase", "FETCH_HEAD")
        except GitError as exc:
            conflicts = _unmerged_paths(vault_root)
            try:
                _git(vault_root, "rebase", "--abort")
            except GitError as abort_exc:
                # Nothing to abort (the rebase never started, e.g. it was
                # refused before touching anything) is the common case. A
                # rebase that DID start and cannot be aborted leaves
                # ``.git/rebase-merge`` behind, which the next write's
                # rebase-in-progress check refuses by name.
                log.info("pull-before-write: rebase --abort: %s", abort_exc)
            if conflicts:
                raise RebaseConflictError(
                    f"rebase onto {remote}/{branch} conflicts on: "
                    + ", ".join(conflicts),
                    conflicts,
                ) from None
            log.warning("pull-before-write: rebase onto %s/%s failed: %s",
                        remote, branch, exc)
            return _blocked_outcome(exc)

        log.info("pull-before-write: rebased onto %d commit(s) from %s/%s",
                 absorbed, remote, branch)
        return RebaseOutcome(fetched=True, absorbed=absorbed, rebased=True)
