"""Connector SDK: idempotent snapshotting, state round-trip, path dispatch.

A fake in-memory connector stands in for a networked source, so these run
offline and deterministically.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from collections.abc import Iterator
from typing import NoReturn

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.connectors import Snapshot, load_state, run_connector, save_state
from ingest_lib.connectors.state import ConnectorState
from ingest_lib.extractors import dispatch_extractor
from ingest_lib.extractors.base import Extractor
from ingest_lib import extractors as _ex
from ingest_lib.pipeline import plan_ingest, run_ingest

# scripts/ has no __init__.py and isn't installed as a package (only
# scripts/ingest_lib is); pin it onto sys.path so pull.py is importable here.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pull  # noqa: E402

_LOG = logging.getLogger("test")
_AT = "2026-07-10T12:00:00Z"


class _FakeConnector:
    name = "fake"

    def __init__(self, items: list[tuple[str, str, bytes]]) -> None:
        # (native_id, filename, payload)
        self._items = items

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]:
        for native_id, filename, payload in self._items:
            yield Snapshot(
                source_class="meetings/fake", native_id=native_id,
                filename=filename, payload=payload,
            )


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def test_pull_writes_snapshots_into_inbox(tmp_path: Path) -> None:
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


def test_repull_unchanged_is_a_noop(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "a.json", b"same-bytes")])
    run_connector(conn, paths, pulled_at=_AT, logger=_LOG)
    dest = paths.root / "inbox/meetings/fake/a.json"
    mtime_before = dest.stat().st_mtime_ns

    stats2 = run_connector(conn, paths, pulled_at="2026-07-11T00:00:00Z", logger=_LOG)
    assert stats2.written == 0 and stats2.skipped == 1
    # The unchanged snapshot was skipped BEFORE touching inbox — untouched.
    assert dest.stat().st_mtime_ns == mtime_before


def test_repull_changed_content_writes_new_snapshot_without_overwrite(tmp_path: Path) -> None:
    # F8: a changed re-pull must never overwrite the inbox snapshot already
    # archived by a prior ingest — it lands as a new, hash-versioned file.
    paths = _vault(tmp_path)
    run_connector(_FakeConnector([("m1", "a.json", b"v1")]), paths, pulled_at=_AT, logger=_LOG)
    stats = run_connector(_FakeConnector([("m1", "a.json", b"v2")]), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    # Old snapshot is untouched.
    assert (paths.root / "inbox/meetings/fake/a.json").read_bytes() == b"v1"
    # New content lands beside it under a versioned name.
    new_snaps = [p for p in stats.snapshots if p != "inbox/meetings/fake/a.json"]
    assert len(new_snaps) == 1
    new_path = paths.root / new_snaps[0]
    assert new_path.read_bytes() == b"v2"
    # State points at the new path for the next pull's idempotency check.
    state = load_state(paths, "fake")
    assert state.entries["m1"]["inbox_path"] == new_snaps[0]


def test_repull_identical_bytes_already_on_disk_reuses_same_path(tmp_path: Path) -> None:
    # Degraded/corrupt state (native id unrecorded) but the same bytes are
    # already sitting at the stable path: no spurious versioned duplicate.
    paths = _vault(tmp_path)
    run_connector(_FakeConnector([("m1", "a.json", b"same")]), paths, pulled_at=_AT, logger=_LOG)
    state_path = paths.metadata / "connectors" / "fake.json"
    state_path.unlink()  # simulate lost/corrupt state
    stats = run_connector(_FakeConnector([("m1", "a.json", b"same")]), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    assert stats.snapshots == ["inbox/meetings/fake/a.json"]
    assert len(list((paths.root / "inbox/meetings/fake").glob("*.json"))) == 1


def test_updated_item_reingests_instead_of_archive_clash(tmp_path: Path) -> None:
    # End-to-end F8 repro: pull v1, ingest it for real, pull a changed v2,
    # then a second ingest must process the new snapshot, not clash.
    paths = _vault(tmp_path)
    _LOG_ = logging.getLogger("repro")
    run_connector(_FakeConnector([("m1", "standup.json", b'{"v":1}')]), paths, pulled_at=_AT, logger=_LOG_)
    plan1 = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG_)
    run_ingest(paths, plan1, dry_run=False, logger=_LOG_)

    run_connector(_FakeConnector([("m1", "standup.json", b'{"v":2}')]), paths, pulled_at=_AT, logger=_LOG_)
    plan2 = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG_)
    stats2 = run_ingest(paths, plan2, dry_run=False, logger=_LOG_)

    assert stats2.manual_review == 0  # no archive-clash
    assert stats2.processed == 1
    index_lines = (paths.metadata / "index.jsonl").read_text().strip().splitlines()
    last = json.loads(index_lines[-1])
    assert last["status"] == "processed"
    assert last.get("extractor") != "archive-clash"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "a.json", b"x")])
    stats = run_connector(conn, paths, pulled_at=_AT, dry_run=True, logger=_LOG)
    assert stats.written == 1  # would-write count
    assert not (paths.root / "inbox/meetings/fake/a.json").exists()
    # No state persisted on a dry run.
    assert not (paths.metadata / "connectors" / "fake.json").exists()


def test_state_round_trips_atomically(tmp_path: Path) -> None:
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


def test_corrupt_state_degrades_to_empty(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    p = paths.metadata / "connectors" / "fake.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")
    state = load_state(paths, "fake")
    assert state.entries == {} and state.cursor is None


def test_dispatch_routes_source_class_prefix_over_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    # A connector snapshot is .json (owned by text.py) but must route to the
    # source-class extractor when its logical path matches the prefix.
    def sentinel(raw_path: Path, dest_dir: Path) -> NoReturn:
        raise NotImplementedError
    sentinel_extractor: Extractor = sentinel
    monkeypatch.setitem(_ex._SOURCE_CLASS_REGISTRY, "meetings/fake/", sentinel_extractor)

    routed = dispatch_extractor(Path("a.json"), relative_path="meetings/fake/a.json")
    assert routed is sentinel
    # A .json elsewhere still uses the suffix map (text extractor), not the prefix.
    other = dispatch_extractor(Path("a.json"), relative_path="notes/a.json")
    assert other is not sentinel and other is not None
    # No relative_path -> pure suffix behaviour (unchanged).
    assert dispatch_extractor(Path("a.json")) is not sentinel


def test_dispatch_ignores_source_class_prefix_for_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AUD-025: the prefix route exists for connector .json snapshots. A PDF
    # or Markdown file a human filed in the same folder must keep its own
    # extractor rather than being handed to the snapshot parser (which would
    # reject it and move it to archive/failed).
    def sentinel(raw_path: Path, dest_dir: Path) -> NoReturn:
        raise NotImplementedError
    sentinel_extractor: Extractor = sentinel
    monkeypatch.setitem(_ex._SOURCE_CLASS_REGISTRY, "meetings/fake/", sentinel_extractor)

    for name in ("a.pdf", "a.md", "a.txt"):
        routed = dispatch_extractor(Path(name), relative_path=f"meetings/fake/{name}")
        assert routed is not sentinel, name
        assert routed is _ex._REGISTRY[Path(name).suffix]
    # An extensionless file under the prefix has no suffix route either.
    assert dispatch_extractor(Path("a"), relative_path="meetings/fake/a") is None
    # The .json snapshot still routes by prefix.
    assert dispatch_extractor(Path("a.json"), relative_path="meetings/fake/a.json") is sentinel


def test_snapshots_and_state_honour_the_umask(tmp_path: Path) -> None:
    # AUD-091: mkstemp hardcodes 0600, so snapshots and connector state came
    # out more restrictive than every other vault file (raw copies are 0644).
    # Vault content a human reads and edits follows the process umask.
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "a.json", b'{"x": 1}')])
    prior = os.umask(0o022)
    try:
        run_connector(conn, paths, pulled_at=_AT, logger=_LOG)
    finally:
        os.umask(prior)
    snap = paths.root / "inbox/meetings/fake/a.json"
    state = paths.metadata / "connectors/fake.json"
    assert snap.stat().st_mode & 0o777 == 0o644
    assert state.stat().st_mode & 0o777 == 0o644


def test_snapshot_and_state_mode_follow_a_restrictive_umask(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    conn = _FakeConnector([("m1", "a.json", b'{"x": 1}')])
    prior = os.umask(0o077)
    try:
        run_connector(conn, paths, pulled_at=_AT, logger=_LOG)
    finally:
        os.umask(prior)
    assert (paths.root / "inbox/meetings/fake/a.json").stat().st_mode & 0o777 == 0o600
    assert (paths.metadata / "connectors/fake.json").stat().st_mode & 0o777 == 0o600


def test_pull_log_stamps_event_time_not_flush_time(tmp_path: Path) -> None:
    # AUD-092: the converter ignored the record's timestamp, so a buffered
    # flush stamped the wrong second (logging_setup.py already fixed this).
    logger = pull._configure_logger(tmp_path, "fake")
    try:
        file_h = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        record = logging.LogRecord("t", logging.INFO, "f", 1, "msg", None, None)
        record.created = 0.0
        assert file_h.formatter is not None
        assert file_h.formatter.format(record).startswith("1970-01-01T00:00:00Z ")
    finally:
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)


# --------------------------------------------------- base.py shared helpers
# AUD-105: meeting_filename and normalize_attendees are reached only through
# the Granola/JustRec connectors, which the fake connector above bypasses —
# so neither ran anywhere in the suite. They are pure functions; test them
# directly rather than behind a networked source.

def test_meeting_filename_is_slugged_dated_and_collision_safe() -> None:
    from ingest_lib.connectors.base import meeting_filename
    assert meeting_filename(
        "2026-07-10", "Weekly Sync: Q3 / plans!", "native-id-a"
    ) == "2026-07-10-weekly-sync-q3-plans-a0e11fcc.json"
    # Same date AND same title, different meeting: the id hash keeps the two
    # filenames apart so neither silently overwrites the other in inbox/.
    same_a = meeting_filename("2026-07-10", "Standup", "native-id-a")
    same_b = meeting_filename("2026-07-10", "Standup", "native-id-b")
    assert same_a != same_b
    assert same_a == "2026-07-10-standup-a0e11fcc.json"
    assert same_b == "2026-07-10-standup-d13dfafd.json"


def test_meeting_filename_falls_back_for_empty_date_and_title() -> None:
    from ingest_lib.connectors.base import meeting_filename
    # A title that slugs to nothing must not produce a leading/doubled dash.
    assert meeting_filename("", "!!!", "native-id-a") == \
        "undated-meeting-a0e11fcc.json"


def test_normalize_attendees_accepts_strings_dicts_and_drops_the_rest() -> None:
    from ingest_lib.connectors.base import normalize_attendees
    assert normalize_attendees([
        "  Ada Lovelace  ",
        {"name": "Grace Hopper", "email": "g@example.com"},
        {"email": "no-name@example.com"},   # no usable name -> dropped
        {"name": "   "},                     # whitespace only -> dropped
        {"name": None},                      # never the string 'None'
        None,
        42,
    ]) == ["Ada Lovelace", "Grace Hopper"]


def test_normalize_attendees_returns_empty_for_a_non_list() -> None:
    from ingest_lib.connectors.base import normalize_attendees
    assert normalize_attendees(None) == []
    assert normalize_attendees({"name": "Ada"}) == []
    assert normalize_attendees("Ada") == []
