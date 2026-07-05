"""Background refresher for derived state after MCP writes.

A write tool's job ends when the note is on disk and committed. The
DERIVED state — embedding index rows, the connection graph, concept
notes, dashboards — must follow, but none of it belongs on the request
path: the first embedding upsert cold-loads sentence-transformers
(~5-15 s of torch init + weights), and a graph rebuild walks the whole
vault. ``IndexRefresher`` moves all of that onto one daemon thread,
mirroring ``push_queue.PushWorker``'s lazy-start/coalesce pattern.

Semantics:

- ``enqueue(rel_path, graph_changed=...)`` is cheap and thread-safe: it
  adds the path to a dirty set and pokes the worker. A burst of writes
  (one MCP call per note) coalesces into ONE batch after a short
  debounce — one embedder call, at most one graph rebuild.
- ``graph_changed`` is sticky per batch: if ANY entry in the batch
  changed topics/relations, the batch ends with a connections +
  concepts (+ dashboards, when that module exists) rebuild, and the
  derived notes that actually changed are committed as
  ``mcp: refresh derived notes`` followed by an async push request.
- Every failure is caught, logged, and audited; the thread never dies
  and a write NEVER depends on reindex success — search freshness
  degrades, vault content does not.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    import numpy as np

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Module (not function) imports so tests can monkeypatch e.g.
# ``semantic.upsert_notes`` and the patch is visible through here.
from ingest_lib import concepts as _concepts  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib import connections as _connections  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib import dashboards as _dashboards  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib import semantic as _semantic  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib.config import paths_for_root  # type: ignore[import-not-found]  # noqa: E402

from .audit import AuditLog
from .git_ops import commit_paths

log = logging.getLogger(__name__)


class IndexRefresher:
    """Single daemon thread that refreshes derived state for dirty notes."""

    def __init__(
        self,
        vault_root: Path,
        *,
        audit: AuditLog,
        enabled: bool = True,
        debounce_seconds: float = 2.0,
        encode: Callable[[list[str]], np.ndarray] | None = None,
        request_push: Callable[[], str] | None = None,
    ) -> None:
        self._vault_root = vault_root
        self._audit = audit
        self._enabled = enabled
        self._debounce = debounce_seconds
        # Test seam: a fake encoder keeps the suite offline. None means
        # the real model, loaded lazily on the worker thread.
        self._encode = encode
        self._request_push = request_push

        self._wake = threading.Event()
        self._stopping = threading.Event()
        # Guards _dirty and _thread (lazy start).
        self._lock = threading.Lock()
        # rel_path -> graph_changed; re-enqueueing a path ORs the flag so
        # a graph-relevant edit can't be downgraded by a later body edit.
        self._dirty: dict[str, bool] = {}
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- API

    def enqueue(self, rel_path: str, *, graph_changed: bool) -> Literal["queued", "off"]:
        """Mark a vault-relative note path dirty. Returns the resulting
        index_refresh state: "off" when refreshing is disabled, else
        "queued"."""
        if not self._enabled:
            return "off"
        with self._lock:
            self._dirty[rel_path] = self._dirty.get(rel_path, False) or graph_changed
            # Lazy thread start: a server that never writes never spawns
            # the thread (and never cold-loads the embedder). Daemon so a
            # hung rebuild can't block interpreter exit.
            if self._thread is None and not self._stopping.is_set():
                self._thread = threading.Thread(
                    target=self._run, name="brain-index-refresher", daemon=True
                )
                self._thread.start()
        self._wake.set()
        return "queued"

    def pending(self) -> int:
        """Number of dirty paths not yet drained into a batch."""
        with self._lock:
            return len(self._dirty)

    def stop(self, flush_seconds: float = 10.0) -> None:
        """Shut down: stop the worker, then best-effort process whatever
        is still dirty so a write made just before shutdown isn't left
        unindexed. Failures degrade to a warning. There is NO automatic
        catch-up (the dirty set is in-memory only) — a periodic or manual
        ``--rebuild-search-index`` is the recovery path, so a dropped batch
        is logged + audited rather than lost silently."""
        self._stopping.set()
        self._wake.set()  # unblock any wait so the thread exits promptly
        thread = self._thread
        if thread is not None:
            thread.join(timeout=flush_seconds)
        # Only drain from THIS thread once the worker is definitely not
        # mid-batch; two concurrent _process_batch calls could race the
        # embeddings writer lock and the git index.
        if thread is None or not thread.is_alive():
            batch = self._drain()
            if batch:
                self._process_batch(batch)
        elif self.pending():
            # Worker still running after the join timeout (e.g. mid cold model
            # load): we can't safely drain here, and nothing catches up on
            # restart. Make the loss visible instead of dropping it silently.
            dropped = self._drain()
            log.warning(
                "reindex: %d dirty path(s) dropped on shutdown (worker still "
                "busy) — run 'ingest.py --rebuild-search-index' to catch up",
                len(dropped),
            )
            self._audit.tool_event(
                agent="system", tool="reindex", path=None, outcome="dropped",
                detail=f"{len(dropped)} path(s) unindexed at shutdown",
            )

    # ----------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            self._wake.wait()
            if self._stopping.is_set():
                return  # stop() best-effort-drains what's left
            self._wake.clear()
            # Debounce: wait out the burst so N rapid writes become one
            # batch. The wait doubles as the stop signal check.
            if self._stopping.wait(self._debounce):
                return
            batch = self._drain()
            if batch:
                self._process_batch(batch)

    def _drain(self) -> dict[str, bool]:
        with self._lock:
            batch = dict(self._dirty)
            self._dirty.clear()
        return batch

    def _process_batch(self, batch: dict[str, bool]) -> None:
        """One batch: embed the dirty notes, then (if any entry touched
        topics/relations) rebuild the graph artefacts and commit exactly
        the derived notes that changed. All failures are swallowed after
        logging + auditing — the worker thread must never die."""
        paths = paths_for_root(self._vault_root)
        try:
            # (a) Incremental embed. First call cold-loads the embedder
            # (~5-15 s) — that cost is exactly WHY this runs here on the
            # background thread and not on the write path.
            _semantic.upsert_notes(
                paths, sorted(batch), logger=log, encode=self._encode
            )
            # (b) Graph inputs changed: derived notes must follow.
            if any(batch.values()):
                self._rebuild_derived(paths)
        except Exception as exc:  # noqa: BLE001 — deliberate: thread must survive
            log.warning("reindex: batch of %d failed: %s", len(batch), exc)
            self._audit.tool_event(
                agent="system",
                tool="reindex",
                path=None,
                outcome="failed",
                detail=f"{type(exc).__name__}: {exc}"[:300],
            )

    def _rebuild_derived(self, paths) -> None:
        conn = _connections.rebuild_connections(paths, logger=log)
        cstats = _concepts.rebuild_concepts(paths, logger=log, related=conn.related)
        derived: list[str] = [*cstats.written_paths, *cstats.removed_paths]

        # Dashboards are a hard dependency (they've existed since that stage
        # landed). Import at module top and call directly: the old
        # importlib + `except ImportError` fallback silently swallowed a
        # genuine ImportError from INSIDE dashboards.py, disabling dashboard
        # refreshes with zero log output. A real breakage now surfaces via
        # _process_batch's broad handler instead of being masked.
        dstats = _dashboards.rebuild_dashboards(paths, logger=log)
        derived.extend(getattr(dstats, "written_paths", ()))

        if not derived:
            return
        # connections.jsonl is gitignored telemetry-adjacent state; only
        # the derived NOTES are vault content worth a commit.
        outcome = commit_paths(
            self._vault_root,
            paths=[self._vault_root / rel for rel in sorted(set(derived))],
            message="mcp: refresh derived notes",
        )
        if outcome.committed and self._request_push is not None:
            self._request_push()
