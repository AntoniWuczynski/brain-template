"""Drive a connector: pull, skip-unchanged, snapshot into ``inbox/``.

The runner is the deterministic half of a connector run — given the same
snapshots and prior state it writes the same files. It never re-writes an
unchanged item (state check) and writes each payload atomically, so an
interrupted pull leaves no torn snapshot in ``inbox/``.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import VaultPaths
from .base import Connector
from .state import load_state, save_state


@dataclass
class PullStats:
    written: int = 0
    skipped: int = 0                       # unchanged since last pull
    snapshots: list[str] = field(default_factory=list)  # inbox relpaths written


def _atomic_write_bytes(dest: Path, payload: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".snap-", suffix=dest.suffix, dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def run_connector(
    connector: Connector,
    paths: VaultPaths,
    *,
    pulled_at: str,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> PullStats:
    """Pull ``connector`` into ``inbox/`` and update its state.

    ``pulled_at`` is the ISO timestamp stamped into the (metadata-only) state
    entries; it is injected rather than read from the clock so runs are
    reproducible in tests. Snapshots whose ``(native_id, content_hash)`` are
    already recorded are skipped before any write.
    """
    log = logger or logging.getLogger(__name__)
    state = load_state(paths, connector.name)
    stats = PullStats()
    for snap in connector.pull(state):
        if state.is_unchanged(snap):
            stats.skipped += 1
            continue
        dest = paths.root / snap.inbox_relpath
        if not dry_run:
            _atomic_write_bytes(dest, snap.payload)
        state.record(snap, pulled_at=pulled_at)
        stats.written += 1
        stats.snapshots.append(snap.inbox_relpath)
        log.info("pull %s: %s", connector.name, snap.inbox_relpath)
    if not dry_run:
        save_state(paths, state)
    log.info(
        "pull %s: written=%d skipped=%d%s",
        connector.name, stats.written, stats.skipped, " (dry-run)" if dry_run else "",
    )
    return stats
