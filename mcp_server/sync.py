"""The one critical section every vault write passes through.

The vault's working tree is shared state with a second writer (the
owner's Obsidian clone, through the remote) and with two of the server's
own background threads (``reindex.IndexRefresher``, which writes and
commits derived notes, and ``push_queue.PushWorker``, which rebases when
a push is rejected). A write tool reads a note, transforms it, writes it
back and commits — and every one of those steps assumes HEAD and the
working tree have not moved underneath it.

``WRITE_LOCK`` is that assumption, made real. Everything that moves HEAD
or leaves an uncommitted tracked file behind holds it:

- the whole audited write path — pull, tool body, commit — in
  ``tools._audited_write``;
- the refresher's derived-note rebuild + commit;
- the push worker's rebase-and-retry (the network push itself stays
  outside: it only reads refs, and holding a write lock across a 15 s
  network round trip would stall every agent).

It is re-entrant because the tool bodies take it again around their own
read-modify-write, and ``entity_tools`` / ``memory_tools`` take it around
theirs. That nesting is harmless and left in place: it documents where
the inner critical section is even though the outer one already covers it.

This module also owns pull-before-write (AUD-122) — the fetch/rebase that
runs at the top of that critical section, and the judgement about what
each of its outcomes means for the write in hand — plus the HEAD guard
that covers what a lock cannot: this vault is a working clone that a
human, a maintenance script (``sweep.py``, ``consolidate.py``, the dream
timer) or an ``ingest.py`` run can commit or rebase in from ANOTHER
process while a tool call is between its read of a note and its write.
"""
from __future__ import annotations

import logging
import posixpath
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import ToolError
from .git_ops import (
    DirtyTreeError,
    GitError,
    RebaseConflictError,
    current_branch,
    head_sha,
    pull_rebase,
    rebase_in_progress,
)

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger(__name__)

WRITE_LOCK = threading.RLock()

# How a human clears a standing rebase conflict. Named in the refusal, the
# audit row and the push worker's status, because none of the three are
# things an agent can retry its way out of.
_MANUAL_RECOVERY = (
    "settle it by hand: resolve the note on the other clone and push, or run "
    "'git rebase' in the server's vault; retrying the write cannot fix it"
)


# The commit each write-tool thread believes it is building on: the HEAD
# recorded once the pull is done, read back before any byte of that thread's
# write reaches disk. Thread-local because the critical section is per-thread
# (WRITE_LOCK is re-entrant) and because a value left behind by one call must
# never be consulted by the next.
_section = threading.local()


class _HeadMark:
    """One nesting level of :func:`write_section`, saved for restore."""

    def __init__(self, root: Path | None, head: str | None) -> None:
        self.root = root
        self.head = head


@contextmanager
def write_section(vault_root: Path) -> Iterator[None]:
    """Mark this thread as building a write on the current HEAD, so
    :func:`refuse_if_head_moved` can tell when that stops being true.

    Saves and restores rather than sets and clears: a value left behind by
    one call must never be read by the next one on the same pooled thread.
    """
    previous = _HeadMark(
        getattr(_section, "root", None), getattr(_section, "head", None)
    )
    _section.root = vault_root
    _section.head = head_sha(vault_root)
    try:
        yield
    finally:
        _section.root = previous.root
        _section.head = previous.head


def refuse_if_head_moved() -> None:
    """Refuse before writing when HEAD is no longer the commit this write
    was prepared against.

    The tool read the note, transformed it in memory and is about to write
    the result back. If HEAD moved in between, that read is stale — a
    rebase or a fast-forward has replaced the note's bytes on disk, and
    writing the transformed text over them destroys whatever came in (the
    IMP-029 loss: an absorbed hand edit overwritten with no conflict, no
    refusal and an ``ok`` audit row).

    ``WRITE_LOCK`` stops the server's own threads from doing this; nothing
    can stop another PROCESS sharing the clone, so this is checked rather
    than assumed. Unlike a conflict, it IS retryable: the retry re-reads
    the note as it now stands.

    A no-op outside a write section (nothing recorded) and when HEAD can't
    be read (a repo with no commits yet) — this hardens a write, it does
    not gate one on git being answerable.
    """
    root = getattr(_section, "root", None)
    expected = getattr(_section, "head", None)
    if root is None or expected is None:
        return
    current = head_sha(root)
    if current is None or current == expected:
        return
    raise ToolError(
        "refusing to write: the vault's HEAD moved while this write was in "
        "flight (another process committed, pulled or rebased in the shared "
        "clone), so the note this write was built from is no longer what is "
        "on disk; nothing was written — re-read the note and retry"
    )


def _path_segments(raw: str) -> tuple[str, ...]:
    """Normalised, extension-stripped path segments of a vault path.

    Used to compare a git-reported path (``knowledge/people/anna.md``)
    against whatever identifier a tool was called with — a vault-relative
    note path, a concept slug (``anna``), an inbox-relative path, a node
    id (``people/anna``). Dropping the extension and comparing segment
    suffixes makes all four line up.
    """
    cleaned = raw.strip().strip("/")
    if not cleaned:
        return ()
    root, _ext = posixpath.splitext(posixpath.normpath(cleaned))
    return tuple(seg for seg in root.split("/") if seg not in ("", "."))


def touches(conflict_paths: tuple[str, ...], write_path: str | None) -> bool:
    """Whether any of ``conflict_paths`` could be what this write targets.

    A rebase replays every unpushed local commit, so its conflicts are
    usually about notes the write in hand never mentions. Only an
    intersection is the write's problem.

    Deliberately generous in both directions: a conflict path matches when
    either identifier's segments are a suffix of the other's, so a concept
    slug matches ``knowledge/concepts/<slug>.md`` and a node id matches its
    note. An unidentifiable target (no path at all) counts as a match —
    unable to prove the write is unrelated, refuse rather than write.
    """
    target = _path_segments(write_path) if write_path else ()
    if not target:
        return True
    for raw in conflict_paths:
        segments = _path_segments(raw)
        if not segments:
            continue
        shorter, longer = (
            (segments, target) if len(segments) <= len(target) else (target, segments)
        )
        if longer[len(longer) - len(shorter):] == shorter:
            return True
    return False


def require_writable_branch(runtime: Runtime) -> None:
    """Refuse BEFORE any disk write when the vault's HEAD is not somewhere
    this server may commit. ``commit_paths`` guards the commit too, but by
    then the file is on disk and the tool would report a
    written-but-uncommitted result; refusing up front keeps the shared
    working tree clean and makes the cause visible to the client.

    Two shapes, and the rebase one is checked first because it explains
    the other: an interrupted rebase parks HEAD detached with a CLEAN
    tree, so the branch message alone would send a reader looking for a
    stray checkout instead of the half-finished rebase that caused it.
    """
    worker = runtime.push_worker
    if rebase_in_progress(worker.vault_root):
        raise ToolError(
            "refusing to write: a rebase is in progress in the vault "
            "(git left rebase-merge/rebase-apply behind), so HEAD is parked "
            "mid-replay and a write now would land off the branch; finish it "
            "or run 'git rebase --abort' in the vault, then retry"
        )
    expected = worker.branch
    current = current_branch(worker.vault_root)
    if current != expected:
        raise ToolError(
            f"refusing to write: the vault has {current!r} checked out, not the "
            f"configured {expected!r}; a write now would be stranded off {expected!r}"
        )


def pull_before_write(
    runtime: Runtime, *, tool: str, path: str | None, agent: str
) -> None:
    """Absorb the other clone's commits before this write lands (AUD-122).

    The server's clone and the owner's Obsidian clone both push to one
    remote, so a write that never fetches leaves the server permanently
    behind after the first hand edit. Fast-forward when this clone has
    nothing of its own, rebase otherwise. Runs under ``WRITE_LOCK``, held
    by the caller through the tool body and its commit.

    Only two things refuse the write:

    - a **dirty tree** — uncommitted changes to tracked files, which under
      this lock can only be a foreign edit in flight that a rebase would
      carry;
    - a **conflict on a path this write targets** — the IMP-029 case. It
      is not retryable and does not pretend to be.

    Everything else is recorded and the write proceeds locally, because
    none of it is the agent's to fix and losing the agent's work to it
    would be worse than being behind:

    - an unreachable remote (the laptop case) — the next push carries both
      sides, exactly as before;
    - a dirty tree (a hand edit in flight in Obsidian or another session —
      the working clone's normal state on the Mac): the pull is skipped for
      this write with a ``pull-skipped`` audit row, the write commits only
      its own paths, and the next clean-tree write absorbs the remote;
    - a conflict on some OTHER note, from an earlier unpushed commit: the
      pull is skipped for this write, the write commits locally, and the
      standing conflict is reported once per write as a ``pull-conflict``
      audit row and through ``PushWorker.status()``;
    - a blocked pull (an untracked file where an incoming commit adds one,
      a broken repo state): a ``pull-failed`` audit row naming the cause,
      and the same status flag — the clone is behind and nothing but a
      human will change that.
    """
    worker = runtime.push_worker
    if not worker.push_enabled:
        # No remote workflow configured: nothing pushes, so nothing to absorb.
        return
    try:
        outcome = pull_rebase(
            worker.vault_root,
            remote=worker.remote,
            branch=worker.branch,
            expected_branch=worker.branch,
        )
    except DirtyTreeError as exc:
        # A rebase needs a clean tree and this one is not: someone is editing
        # a tracked file outside this process (Obsidian on the Mac, another
        # session). That is the working clone's NORMAL state before the
        # migration, so it must not block the agent's write — the write only
        # stages its own paths and commits, leaving the hand edit exactly as
        # it was, which is how this server behaved before it pulled at all.
        # What is lost is only that this write did not absorb the remote;
        # said once per write, and cleared by the next clean-tree write.
        _report_stuck(
            runtime, tool=tool, path=path, agent=agent, outcome="pull-skipped",
            detail=(
                "pull-before-write skipped: uncommitted changes to "
                + ", ".join(exc.paths)
                + " (a hand edit in flight) — the write committed locally and "
                "will absorb the remote once the tree is clean"
            ),
        )
        return
    except RebaseConflictError as exc:
        if touches(exc.paths, path):
            raise ToolError(
                "refusing to write: the remote has conflicting edits to "
                f"{', '.join(exc.paths)}, which this write targets; the rebase "
                f"was aborted and nothing changed — {_MANUAL_RECOVERY}"
            ) from None
        _report_stuck(
            runtime, tool=tool, path=path, agent=agent, outcome="pull-conflict",
            detail=(
                "pull-before-write skipped: rebasing onto the remote conflicts on "
                + ", ".join(exc.paths)
                + " — paths this write does not touch, so it committed locally. "
                "The clone cannot absorb the remote until that is settled: "
                + _MANUAL_RECOVERY
            ),
        )
        return
    except GitError as exc:
        # The branch guard, or anything else git refused outright.
        _report_stuck(
            runtime, tool=tool, path=path, agent=agent, outcome="pull-failed",
            detail=f"pull-before-write could not run: {str(exc)[:200]}",
        )
        return
    if outcome.blocked:
        _report_stuck(
            runtime, tool=tool, path=path, agent=agent, outcome="pull-failed",
            detail=(
                "pull-before-write reached the remote but could not absorb it: "
                f"{outcome.blocked} — the write committed locally and this clone "
                "stays behind until that is cleared by hand"
            ),
        )
        return
    if outcome.fetched:
        # The remote is reachable and its commits are in: whatever was
        # blocking convergence no longer is.
        worker.note_sync_problem(None)
    if outcome.absorbed:
        runtime.audit.tool_event(
            agent=agent, tool=tool, path=path, outcome="rebased",
            detail=f"pull-before-write absorbed {outcome.absorbed} commit(s) by "
                   f"{'rebase' if outcome.rebased else 'fast-forward'}",
        )


def _report_stuck(
    runtime: Runtime, *, tool: str, path: str | None, agent: str,
    outcome: str, detail: str,
) -> None:
    """Record a pull that did not happen, in the two places a human or an
    agent looks: the audit log (durable, greppable, the reason AUD-122
    exists) and the push worker's status (live). The server log gets it at
    error level too — a clone that has stopped converging is not routine."""
    log.error("%s (%s): %s", outcome, tool, detail)
    runtime.audit.tool_event(
        agent=agent, tool=tool, path=path, outcome=outcome, detail=detail[:500],
    )
    runtime.push_worker.note_sync_problem(detail[:500])
