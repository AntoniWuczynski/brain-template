"""Drive a connector: pull, skip-unchanged, snapshot into ``inbox/``.

The runner is the deterministic half of a connector run — given the same
snapshots and prior state it writes the same files. It never re-writes an
unchanged item (state check) and writes each payload atomically, so an
interrupted pull leaves no torn snapshot in ``inbox/``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..atomic import atomic_write_bytes, umask_mode
from ..config import VaultPaths
from .base import Connector, Snapshot
from .state import load_state, save_state


@dataclass
class PullStats:
    written: int = 0
    skipped: int = 0                       # unchanged since last pull
    snapshots: list[str] = field(default_factory=list)  # inbox relpaths written


def _atomic_write_bytes(dest: Path, payload: bytes) -> None:
    # A snapshot is vault content a human reads and edits, so it gets the
    # umask's mode rather than mkstemp's 0600.
    atomic_write_bytes(
        dest, payload, prefix=".snap-", suffix=dest.suffix, mode=umask_mode()
    )


def _versioned_relpath(relpath: str, content_hash: str) -> str:
    """Insert a hash tag before the extension, e.g. ``a.json`` ->
    ``a.3f9c1a7b2e04.json`` — two versions of the same item are told apart
    by this content-hash segment in the filename."""
    p = Path(relpath)
    return str(p.with_name(f"{p.stem}.{content_hash[:12]}{p.suffix}"))


def _resolve_dest(paths: VaultPaths, snap: Snapshot) -> tuple[Path, str]:
    """Pick the inbox path to write ``snap`` to.

    The connector's own (stable) ``inbox_relpath`` is used when it is free or
    already holds byte-identical content; otherwise a changed re-pull of the
    same native id is routed to a hash-versioned sibling path so the prior
    snapshot already sitting in ``inbox/`` is never overwritten.
    """
    relpath = snap.inbox_relpath
    dest = paths.root / relpath
    if dest.exists():
        try:
            unchanged_on_disk = dest.read_bytes() == snap.payload
        except OSError:
            unchanged_on_disk = False
        if not unchanged_on_disk:
            relpath = _versioned_relpath(relpath, snap.content_hash)
            dest = paths.root / relpath
    return dest, relpath


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
        dest, relpath = _resolve_dest(paths, snap)
        if not dry_run:
            _atomic_write_bytes(dest, snap.payload)
        state.record(snap, pulled_at=pulled_at, inbox_relpath=relpath)
        stats.written += 1
        stats.snapshots.append(relpath)
        log.info("pull %s: %s", connector.name, relpath)
    if not dry_run:
        save_state(paths, state)
    log.info(
        "pull %s: written=%d skipped=%d%s",
        connector.name, stats.written, stats.skipped, " (dry-run)" if dry_run else "",
    )
    return stats
