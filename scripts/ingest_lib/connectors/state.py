"""Per-connector pull state at ``metadata/connectors/<name>.json``.

Records what each connector has already snapshotted so a re-pull of an
unchanged item is a no-op *before* it touches ``inbox/`` — the same
idempotency contract the ingest pipeline holds via ``index.jsonl``. Written
atomically (temp + fsync + rename), like every other metadata file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..atomic import atomic_write_text, umask_mode
from ..config import VaultPaths
from .base import Snapshot


@dataclass
class ConnectorState:
    """Mutable pull state for one connector.

    ``entries`` maps ``native_id -> {content_hash, inbox_path, pulled_at}``.
    ``cursor`` is an opaque, connector-defined incremental marker (e.g. a
    Zotero ``Last-Modified-Version``); the runner never interprets it.
    """

    name: str
    entries: dict[str, dict[str, str]] = field(default_factory=dict)
    cursor: str | None = None

    def is_unchanged(self, snap: Snapshot) -> bool:
        prior = self.entries.get(snap.native_id)
        return prior is not None and prior.get("content_hash") == snap.content_hash

    def record(self, snap: Snapshot, *, pulled_at: str, inbox_relpath: str | None = None) -> None:
        """Record ``snap`` as pulled. ``inbox_relpath`` overrides the
        connector's stable ``snap.inbox_relpath`` when the runner routed a
        changed re-pull to a versioned sibling path instead."""
        self.entries[snap.native_id] = {
            "content_hash": snap.content_hash,
            "inbox_path": inbox_relpath if inbox_relpath is not None else snap.inbox_relpath,
            "pulled_at": pulled_at,
        }


def _state_path(paths: VaultPaths, name: str) -> Path:
    return paths.metadata / "connectors" / f"{name}.json"


def load_state(paths: VaultPaths, name: str) -> ConnectorState:
    """Load a connector's state, or an empty one if it has never run. A
    corrupt/unreadable state file degrades to empty rather than crashing the
    pull — worst case is re-snapshotting unchanged items (still idempotent
    downstream via the archive hash)."""
    path = _state_path(paths, name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ConnectorState(name=name)
    if not isinstance(raw, dict):
        return ConnectorState(name=name)
    entries = raw.get("entries")
    cursor = raw.get("cursor")
    return ConnectorState(
        name=name,
        entries=entries if isinstance(entries, dict) else {},
        cursor=cursor if isinstance(cursor, str) else None,
    )


def save_state(paths: VaultPaths, state: ConnectorState) -> None:
    body = json.dumps(
        {"name": state.name, "cursor": state.cursor, "entries": state.entries},
        ensure_ascii=False, indent=2, sort_keys=True,
    )
    # State is vault content under version control alongside 0644 files, so
    # honour the umask rather than keeping mkstemp's 0600.
    atomic_write_text(
        _state_path(paths, state.name), body,
        prefix=f".{state.name}-", suffix=".json", mode=umask_mode(),
    )
