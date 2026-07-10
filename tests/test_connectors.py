"""Connector SDK: idempotent snapshotting, state round-trip, path dispatch.

A fake in-memory connector stands in for a networked source, so these run
offline and deterministically.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ingest_lib.config import paths_for_root
from ingest_lib.connectors import Snapshot, load_state, run_connector, save_state
from ingest_lib.connectors.state import ConnectorState
from ingest_lib.extractors import dispatch_extractor
from ingest_lib import extractors as _ex

_LOG = logging.getLogger("test")
_AT = "2026-07-10T12:00:00Z"


class _FakeConnector:
    name = "fake"

    def __init__(self, items: list[tuple[str, str, bytes]]):
        # (native_id, filename, payload)
        self._items = items

    def pull(self, state):
        for native_id, filename, payload in self._items:
            yield Snapshot(
                source_class="meetings/fake", native_id=native_id,
                filename=filename, payload=payload,
            )


def _vault(tmp_path: Path):
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def test_pull_writes_snapshots_into_inbox(tmp_path: Path):
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "2026-07-10-standup.json", b'{"id": "m1"}')])

    stats = run_connector(conn, paths, pulled_at=_AT, logger=_LOG)

    assert stats.written == 1 and stats.skipped == 0
    dest = paths.root / "inbox/meetings/fake/2026-07-10-standup.json"
    assert dest.read_bytes() == b'{"id": "m1"}'
    # State records the native id -> content hash for next time.
    state = load_state(paths, "fake")
    assert "m1" in state.entries
    assert state.entries["m1"]["inbox_path"] == "inbox/meetings/fake/2026-07-10-standup.json"


def test_repull_unchanged_is_a_noop(tmp_path: Path):
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "a.json", b"same-bytes")])
    run_connector(conn, paths, pulled_at=_AT, logger=_LOG)
    dest = paths.root / "inbox/meetings/fake/a.json"
    mtime_before = dest.stat().st_mtime_ns

    stats2 = run_connector(conn, paths, pulled_at="2026-07-11T00:00:00Z", logger=_LOG)
    assert stats2.written == 0 and stats2.skipped == 1
    # The unchanged snapshot was skipped BEFORE touching inbox — untouched.
    assert dest.stat().st_mtime_ns == mtime_before


def test_repull_changed_content_rewrites(tmp_path: Path):
    paths = _vault(tmp_path)
    run_connector(_FakeConnector([("m1", "a.json", b"v1")]), paths, pulled_at=_AT, logger=_LOG)
    stats = run_connector(_FakeConnector([("m1", "a.json", b"v2")]), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    assert (paths.root / "inbox/meetings/fake/a.json").read_bytes() == b"v2"


def test_dry_run_writes_nothing(tmp_path: Path):
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "a.json", b"x")])
    stats = run_connector(conn, paths, pulled_at=_AT, dry_run=True, logger=_LOG)
    assert stats.written == 1  # would-write count
    assert not (paths.root / "inbox/meetings/fake/a.json").exists()
    # No state persisted on a dry run.
    assert not (paths.metadata / "connectors" / "fake.json").exists()


def test_state_round_trips_atomically(tmp_path: Path):
    paths = _vault(tmp_path)
    state = ConnectorState(name="fake", cursor="v42")
    state.entries["m1"] = {"content_hash": "h", "inbox_path": "inbox/x", "pulled_at": _AT}
    save_state(paths, state)

    reloaded = load_state(paths, "fake")
    assert reloaded.cursor == "v42"
    assert reloaded.entries["m1"]["content_hash"] == "h"
    # Valid JSON on disk (atomic write left no temp file).
    on_disk = json.loads((paths.metadata / "connectors" / "fake.json").read_text())
    assert on_disk["cursor"] == "v42"
    assert not list((paths.metadata / "connectors").glob(".fake-*"))


def test_corrupt_state_degrades_to_empty(tmp_path: Path):
    paths = _vault(tmp_path)
    p = paths.metadata / "connectors" / "fake.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")
    state = load_state(paths, "fake")
    assert state.entries == {} and state.cursor is None


def test_dispatch_routes_source_class_prefix_over_suffix(monkeypatch):
    # A connector snapshot is .json (owned by text.py) but must route to the
    # source-class extractor when its logical path matches the prefix.
    sentinel = object()
    monkeypatch.setitem(_ex._SOURCE_CLASS_REGISTRY, "meetings/fake/", sentinel)

    routed = dispatch_extractor(Path("a.json"), relative_path="meetings/fake/a.json")
    assert routed is sentinel
    # A .json elsewhere still uses the suffix map (text extractor), not the prefix.
    other = dispatch_extractor(Path("a.json"), relative_path="notes/a.json")
    assert other is not sentinel and other is not None
    # No relative_path -> pure suffix behaviour (unchanged).
    assert dispatch_extractor(Path("a.json")) is not sentinel
