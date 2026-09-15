"""Asynchronous, coalescing git push worker.

Write tools commit synchronously (the commit must land before the tool
returns, so the agent's "saved" claim is true) but the push to the
remote is latency we don't need on the request path: a black-holed
network would add up to 15s to every write. PushWorker moves the push
onto a single background thread.

Semantics:

- ``request_push()`` is cheap and idempotent-ish: it pokes an Event.
  N requests during one in-flight push coalesce into exactly one
  follow-up push — git pushes *everything* on the branch, so one push
  covers all commits made since the last one.
- Construction requests one push: the worker is built once per server
  start, and a crash (or a final flush that timed out) can leave commits
  on the branch that nothing else would ever push — writes are the only
  other trigger. It is a no-op against an already-current remote.
- On failure the worker retries on a capped backoff schedule until the
  push succeeds or a fresh request resets the backoff (a new commit is
  a good reason to try again immediately).
- A push the remote rejects as non-fast-forward (the other clone got
  there first) is not a plain failure: the worker fetches, rebases the
  branch the same way pull-before-write does, and retries ONCE. A
  conflict is never resolved — it is logged and left in ``status()``.
  That rebase runs under ``sync.WRITE_LOCK`` (it moves HEAD, so it must
  not land between a write tool's read of a note and its commit) and is
  guarded by ``expected_branch`` (it rewrites history, so it must never
  touch a branch that is merely what happens to be checked out).
- Pushing does NOT take ``git_ops._GIT_LOCK`` or ``sync.WRITE_LOCK``:
  push only reads refs,
  never the index or working tree, so racing a concurrent commit can't
  corrupt anything — worst case the push misses that commit and the
  next push catches up.
- Full git stderr (remote URLs, ssh hints) stays in the server log;
  ``status()`` exposes only a sanitized error string.
- A *run* of failures (and the recovery that ends it, and commits left
  unpushed at shutdown) is written to the audit log as well, so a vault
  that has quietly stopped being backed up is visible to a human or an
  agent reading ``logs/mcp-audit.jsonl`` — not only to whoever tails the
  server log. Pushing still never fails a write.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TypedDict

from .audit import AuditLog
from .git_ops import GitError, RebaseConflictError, _git, pull_rebase
from .sync import WRITE_LOCK

log = logging.getLogger(__name__)

# Upper bound on a single `git push` so a hung remote can't wedge the worker.
_PUSH_TIMEOUT_S = 15.0

# Consecutive failures before the run is escalated from a per-attempt log
# warning to an operator-visible alarm. A push that fails once is normal
# (laptop off the network); one that keeps failing means the vault is
# committing locally and reaching no remote, which nothing else reports:
# status() is not exposed by any tool. With the default retry schedule this
# is ~1.5 minutes of a vault that is no longer backed up.
_ALARM_AFTER_FAILURES = 3


def _is_non_fast_forward(exc: BaseException) -> bool:
    """Whether a failed push was rejected for being behind the remote.

    git has no exit code for it, so this reads the message ``_git`` carried
    out of stderr. Both wordings appear depending on git version and remote
    type; matching too broadly only costs one extra fetch, matching too
    narrowly costs a vault that never pushes again.
    """
    text = str(exc).lower()
    return "non-fast-forward" in text or "fetch first" in text


class PushStatus(TypedDict):
    """Shape of :meth:`PushWorker.status` snapshots."""

    state: str
    consecutive_failures: int
    last_error: str | None
    # A condition that stops this clone converging with the remote and that
    # no retry will clear — a standing rebase conflict, a blocked pull. Set
    # by whoever hits it (the write path's pull, this worker's rebase),
    # cleared by the next successful pull or push. None when the clone is
    # converging normally, which is NOT the same as ``last_error is None``:
    # a push can be failing for a network reason and still be recoverable.
    sync_error: str | None


class PushWorker:
    """Single daemon thread that pushes the vault branch to its remote."""

    def __init__(
        self,
        vault_root: Path,
        *,
        remote: str,
        branch: str,
        enabled: bool,
        retry_schedule: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0),
    ) -> None:
        self._vault_root = vault_root
        self._remote = remote
        self._branch = branch
        self._enabled = enabled
        self._retry_schedule = retry_schedule

        # _wake doubles as the "a push is owed" flag: set by request_push,
        # cleared by the worker right before it pushes, so anything set
        # during a push triggers exactly one follow-up.
        self._wake = threading.Event()
        self._stopping = threading.Event()
        # Guards _thread (lazy start), _state and the failure counters.
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: str = "idle"          # "idle" | "pending" | "retrying"
        self._consecutive_failures: int = 0
        self._last_error: str | None = None
        self._sync_error: str | None = None
        # True once this run of failures has been alarmed, so a long outage
        # writes one audit row rather than one per retry.
        self._alarmed = False
        # Own appender rather than a constructor argument: AuditLog needs
        # nothing but the vault root, and the worker must alarm in
        # production without every caller having to remember to wire it.
        self._audit = AuditLog(vault_root)

        # Push on start, so commits stranded by a crash reach the remote
        # without waiting for the next write — the recovery stop() already
        # documents ("push on next server start"). Off when pushing is
        # disabled, and still off the request path (background thread).
        self.request_push()

    # ------------------------------------------------------------- API


    @property
    def vault_root(self) -> Path:
        return self._vault_root

    @property
    def branch(self) -> str:
        """The branch commits are expected on and pushes target."""
        return self._branch

    @property
    def remote(self) -> str:
        """The remote pushes target and pull-before-write fetches from."""
        return self._remote

    @property
    def push_enabled(self) -> bool:
        """Whether this vault syncs with a remote at all. False means a
        local-only vault: nothing pushes, so nothing has to be pulled
        before a write either."""
        return self._enabled

    def request_push(self) -> str:
        """Ask for a push soon. Returns the resulting push_state:
        "disabled" when pushing is off, else "queued"."""
        if not self._enabled:
            return "disabled"
        with self._lock:
            # Lazy thread start: a server that never writes never spawns
            # the thread. Daemon so a hung push can't block interpreter
            # exit (stop() does an orderly flush when the caller cares).
            if self._thread is None and not self._stopping.is_set():
                self._thread = threading.Thread(
                    target=self._run, name="brain-push-worker", daemon=True
                )
                self._thread.start()
            self._wake.set()
            if self._state == "idle":
                self._state = "pending"
        return "queued"

    def status(self) -> PushStatus:
        """Snapshot for a health/status tool. ``last_error`` is sanitized —
        never contains remote URLs or raw git stderr."""
        with self._lock:
            return {
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "last_error": self._last_error,
                "sync_error": self._sync_error,
            }

    def note_sync_problem(self, detail: str | None) -> None:
        """Record (``detail``) or clear (``None``) the standing reason this
        clone cannot converge with its remote.

        Called from the write path's pull (``sync.pull_before_write``) and
        from this worker's own rebase. Kept separate from the push failure
        counters on purpose: a failing push is usually the network and
        heals itself, whereas this is a human's to settle and would
        otherwise be visible only to whoever tails the server log."""
        with self._lock:
            self._sync_error = detail

    def stop(self, flush_seconds: float = 5.0) -> None:
        """Shut down: stop the worker, then make one final best-effort
        push (short timeout) if anything is still owed. Used on server
        shutdown so the last write of a session isn't stranded locally."""
        self._stopping.set()
        self._wake.set()  # unblock any wait so the thread exits promptly
        thread = self._thread
        if thread is not None:
            thread.join(timeout=flush_seconds)
        with self._lock:
            owed = self._enabled and self._state != "idle"
        if owed:
            try:
                _git(self._vault_root, "push", self._remote, self._branch,
                     timeout=max(0.1, min(flush_seconds, _PUSH_TIMEOUT_S)))
                self._note_success()
                with self._lock:
                    self._state = "idle"
            except GitError as exc:
                # Best-effort only: commits are safe locally and push on
                # next server start / next write — but audit it, so the
                # commits that went to bed on this box are recoverable
                # knowledge and not just a line in the server log.
                log.warning("push worker: final flush push failed: %s", exc)
                self._audit.tool_event(
                    agent="system", tool="push", path=None, outcome="unpushed",
                    detail="final flush push failed at shutdown; "
                           "commits remain local until the next push",
                )

    # ----------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            self._wake.wait()
            if self._stopping.is_set():
                return
            self._wake.clear()
            with self._lock:
                self._state = "pending"

            attempt = 0
            while not self._try_push():
                with self._lock:
                    self._state = "retrying"
                # Capped backoff: walk the schedule, then stay at its tail.
                delay = self._retry_schedule[min(attempt, len(self._retry_schedule) - 1)]
                attempt += 1
                fresh_request = self._wake.wait(timeout=delay)
                if self._stopping.is_set():
                    return
                if fresh_request:
                    # A new commit arrived: reset the backoff and retry
                    # now — one push will carry it and the failed ones.
                    self._wake.clear()
                    attempt = 0

            self._note_success()
            with self._lock:
                # If a request landed mid-push, the outer loop will run
                # again immediately; reflect that instead of "idle".
                self._state = "pending" if self._wake.is_set() else "idle"

    def _note_success(self) -> None:
        """Clear the failure counters. If this ends an alarmed run, close it
        out in the audit log so a reader can tell a vault that recovered from
        one that is still not reaching its remote."""
        with self._lock:
            recovered = self._alarmed
            self._alarmed = False
            self._consecutive_failures = 0
            self._last_error = None
            # The branch reached the remote, so nothing is standing between
            # the two clones any more.
            self._sync_error = None
        if recovered:
            log.info("push worker: push to %s/%s recovered", self._remote, self._branch)
            self._audit.tool_event(
                agent="system", tool="push", path=None, outcome="recovered",
                detail="git push succeeded after a run of failures",
            )

    def _push(self) -> None:
        """The push itself. Deliberately NOT under git_ops._GIT_LOCK — see
        the module docstring."""
        _git(self._vault_root, "push", self._remote, self._branch,
             timeout=_PUSH_TIMEOUT_S)

    def _rebase_and_retry_once(self) -> str | None:
        """Handle a push the remote rejected as non-fast-forward: absorb its
        commits and push once more (AUD-122). Returns None when the retry
        succeeded, else a sanitized error for ``status()``.

        Exactly one retry — a second rejection means another writer is
        pushing faster than this worker, and the normal backoff loop is the
        right place for that. A conflict is never resolved (IMP-029): the
        rebase is aborted, the branch is left where it was, and the conflict
        surfaces through PushStatus for a human to settle.

        The REBASE runs under ``sync.WRITE_LOCK`` — it moves HEAD and the
        working tree, and a write tool between its read of a note and its
        commit cannot survive that. The push either side of it does not:
        it only reads refs, and a 15 s network round trip is not something
        to hold every agent's writes behind.
        """
        log.info("push worker: push to %s/%s was rejected as non-fast-forward; "
                 "rebasing onto the remote", self._remote, self._branch)
        try:
            with WRITE_LOCK:
                outcome = pull_rebase(
                    self._vault_root, remote=self._remote, branch=self._branch,
                    expected_branch=self._branch,
                )
        except RebaseConflictError as exc:
            log.error("push worker: rebase after a rejected push conflicts on %s; "
                      "the branch is unchanged and these commits are NOT pushed",
                      ", ".join(exc.paths))
            sanitized = "git push rejected; rebase conflicts on " + ", ".join(exc.paths)
            self.note_sync_problem(
                sanitized + " — resolve the note on the other clone and push, or "
                "run 'git rebase' in the server's vault; no retry clears it"
            )
            return sanitized
        except GitError as exc:
            # Includes the branch guard: HEAD is not on the configured
            # branch (a debugging checkout, a detached HEAD, an interrupted
            # rebase), so nothing was touched — rewriting whatever branch
            # happens to be out is exactly the damage the guard prevents.
            log.warning("push worker: rebase after a rejected push refused: %s", exc)
            self.note_sync_problem(f"git push rejected and the rebase was refused: {exc}"[:300])
            return "git push rejected and the rebase was refused (details in server log)"
        except Exception as exc:  # noqa: BLE001 — thread must survive any failure
            log.warning("push worker: rebase after a rejected push failed: %s", exc)
            return "git push rejected and the rebase failed (details in server log)"
        if not outcome.fetched:
            return "git push rejected and the fetch failed (details in server log)"
        if outcome.blocked:
            log.error("push worker: the remote's commits could not be absorbed: %s",
                      outcome.blocked)
            self.note_sync_problem(
                "git push rejected and the remote's commits could not be absorbed: "
                f"{outcome.blocked}"
            )
            return "git push rejected; absorbing the remote is blocked (details in server log)"
        try:
            self._push()
        except Exception as exc:  # noqa: BLE001 — thread must survive any failure
            log.warning("push worker: retry push after rebase failed: %s", exc)
            return "git push failed after rebase (details in server log)"
        log.info("push worker: absorbed %d commit(s) from %s/%s and pushed",
                 outcome.absorbed, self._remote, self._branch)
        return None

    def _try_push(self) -> bool:
        """One push attempt. Records a sanitized error on failure; the
        real stderr (remote URL / ssh hints) goes to the server log only.

        A rejection for being behind the remote is retried once through
        ``_rebase_and_retry_once``; every other failure goes straight to
        the backoff loop.

        Catches BOTH GitError and any other exception (e.g. an OSError from
        subprocess spawn under EMFILE/ENOMEM). Letting a non-GitError escape
        would unwind _run and kill the sole daemon thread permanently —
        every later commit would then land locally and never push, while
        status() kept reporting success. Failing soft keeps the retry loop
        (and the visible failure counter) alive."""
        try:
            self._push()
            return True
        except Exception as exc:  # noqa: BLE001 — thread must survive any failure
            if _is_non_fast_forward(exc):
                sanitized_or_none = self._rebase_and_retry_once()
                if sanitized_or_none is None:
                    return True
                sanitized = sanitized_or_none
            else:
                log.warning("push worker: push to %s/%s failed: %s",
                            self._remote, self._branch, exc)
                sanitized = (
                    "git push timed out" if "timed out" in str(exc)
                    else "git push failed (details in server log)"
                )
            with self._lock:
                self._consecutive_failures += 1
                self._last_error = sanitized
                failures = self._consecutive_failures
                alarm = failures >= _ALARM_AFTER_FAILURES and not self._alarmed
                if alarm:
                    self._alarmed = True
            if alarm:
                # Outside the lock: the audit append touches the filesystem.
                log.error(
                    "push worker: %d consecutive pushes to %s/%s failed — commits "
                    "are landing locally and NOT reaching the remote",
                    failures, self._remote, self._branch,
                )
                self._audit.tool_event(
                    agent="system", tool="push", path=None, outcome="failing",
                    detail=f"{failures} consecutive git push failures; vault commits "
                           "are not reaching the remote (details in server log)",
                )
            return False
