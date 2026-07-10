"""Per-connector pull state at ``metadata/connectors/<name>.json``.

Records what each connector has already snapshotted so a re-pull of an
unchanged item is a no-op *before* it touches ``inbox/`` — the same
idempotency contract the ingest pipeline holds via ``index.jsonl``. Written
atomically (temp + fsync + rename), like every other metadata file.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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

    def record(self, snap: Snapshot, *, pulled_at: str) -> None:
        self.entries[snap.native_id] = {
            "content_hash": snap.content_hash,
            "inbox_path": snap.inbox_relpath,
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
    path = _state_path(paths, state.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {"name": state.name, "cursor": state.cursor, "entries": state.entries},
        ensure_ascii=False, indent=2, sort_keys=True,
    )
    fd, tmp = tempfile.mkstemp(prefix=f".{state.name}-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
