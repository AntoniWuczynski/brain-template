"""Meeting connector (Granola + justREC) + the shared meeting extractor."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ingest_lib.config import paths_for_root
from ingest_lib.connectors import CONNECTORS, run_connector
from ingest_lib.extractors import dispatch_extractor
from ingest_lib.extractors import meeting as meeting_ex

_LOG = logging.getLogger("test")
_AT = "2026-07-12T00:00:00Z"


# ------------------------------------------------------------ extractor

def _snapshot(**kw) -> bytes:
    return json.dumps(kw).encode("utf-8")


def test_meeting_extractor_renders_note(tmp_path: Path):
    src = tmp_path / "m.json"
    src.write_bytes(_snapshot(
        connector="granola", id="m1", title="Kern weekly", date="2026-07-12",
        attendees=["Alice Smith", "Bob Jones"],
        summary="Agreed to ship the connector.", transcript="Alice: hello ...",
    ))
    res = meeting_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "# Kern weekly" in res.markdown
    assert "[[knowledge/people/alice-smith]]" in res.markdown
    assert "[[knowledge/people/bob-jones]]" in res.markdown
    assert "Agreed to ship the connector." in res.markdown
    assert "Alice: hello" in res.markdown


def test_meeting_without_body_is_partial(tmp_path: Path):
    src = tmp_path / "m.json"
    src.write_bytes(_snapshot(connector="justrec", id="m2", title="Standup",
                              attendees=["Alice"]))
    res = meeting_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"                 # metadata only, no invented body
    assert "_(no transcript captured)_" in res.markdown


def test_meeting_bad_json_is_manual_review(tmp_path: Path):
    src = tmp_path / "m.json"
    src.write_bytes(b"{ not json")
    res = meeting_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_meeting_snapshot_routes_by_source_class():
    routed = dispatch_extractor(Path("x.json"), relative_path="meetings/granola/2026-07-12-kern.json")
    assert routed is meeting_ex.extract
    routed2 = dispatch_extractor(Path("x.json"), relative_path="meetings/justrec/a.json")
    assert routed2 is meeting_ex.extract


# ------------------------------------------------------------ connectors

def _vault(tmp_path: Path):
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def test_granola_connector_pulls_snapshots(tmp_path: Path, monkeypatch):
    from ingest_lib.connectors import granola
    monkeypatch.setenv("GRANOLA_API_KEY", "test-key")
    monkeypatch.setattr(granola, "_fetch_meetings", lambda _k: [
        {"id": "g1", "title": "Kern sync", "date": "2026-07-12",
         "attendees": [{"name": "Alice"}], "summary": "s", "transcript": "t"},
        {"id": "", "title": "no id — dropped"},
    ])
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["granola"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/meetings/granola").glob("2026-07-12-kern-sync-*.json"))
    assert len(snaps) == 1
    data = json.loads(snaps[0].read_text())
    assert data["connector"] == "granola" and data["attendees"] == ["Alice"]


def test_granola_same_day_same_title_do_not_collide(tmp_path: Path, monkeypatch):
    from ingest_lib.connectors import granola
    monkeypatch.setenv("GRANOLA_API_KEY", "k")
    monkeypatch.setattr(granola, "_fetch_meetings", lambda _k: [
        {"id": "a", "title": "Standup", "date": "2026-07-12", "summary": "one"},
        {"id": "b", "title": "Standup", "date": "2026-07-12", "summary": "two"},
    ])
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["granola"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 2   # both survive — distinct id-hashed filenames
    assert len(list((paths.root / "inbox/meetings/granola").glob("*.json"))) == 2


def test_granola_fetch_error_degrades(tmp_path: Path, monkeypatch):
    from ingest_lib.connectors import granola
    monkeypatch.setenv("GRANOLA_API_KEY", "k")

    def _boom(_k):
        raise OSError("network down")
    monkeypatch.setattr(granola, "_fetch_meetings", _boom)
    stats = run_connector(CONNECTORS["granola"](), _vault(tmp_path), pulled_at=_AT, logger=_LOG)
    assert stats.written == 0   # degraded, not crashed


def test_granola_attendee_without_name_dropped(tmp_path: Path, monkeypatch):
    from ingest_lib.connectors import granola
    monkeypatch.setenv("GRANOLA_API_KEY", "k")
    monkeypatch.setattr(granola, "_fetch_meetings", lambda _k: [
        {"id": "a", "title": "M", "date": "2026-07-12",
         "attendees": [{"email": "x@y.z"}, {"name": "Real"}, None, "Str"]},
    ])
    paths = _vault(tmp_path)
    run_connector(CONNECTORS["granola"](), paths, pulled_at=_AT, logger=_LOG)
    data = json.loads(next((paths.root / "inbox/meetings/granola").glob("*.json")).read_text())
    assert data["attendees"] == ["Real", "Str"]   # no "None", no name-less dict


def test_granola_no_key_pulls_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GRANOLA_API_KEY", raising=False)
    stats = run_connector(CONNECTORS["granola"](), _vault(tmp_path), pulled_at=_AT, logger=_LOG)
    assert stats.written == 0 and stats.skipped == 0


def test_justrec_connector_reads_local_folder(tmp_path: Path, monkeypatch):
    rec_dir = tmp_path / "justrec-out"
    rec_dir.mkdir()
    (rec_dir / "meeting1.json").write_text(json.dumps({
        "title": "Client call", "date": "2026-07-11",
        "participants": ["Carol"], "summary": "notes", "transcript": "hello",
    }), encoding="utf-8")
    monkeypatch.setenv("BRAIN_JUSTREC_DIR", str(rec_dir))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["justrec"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/meetings/justrec").glob("2026-07-11-client-call-*.json"))
    assert len(snaps) == 1
    assert json.loads(snaps[0].read_text())["connector"] == "justrec"


def test_justrec_same_stem_different_subdirs_are_distinct(tmp_path: Path, monkeypatch):
    rec_dir = tmp_path / "rec"
    (rec_dir / "a").mkdir(parents=True)
    (rec_dir / "b").mkdir(parents=True)
    for sub in ("a", "b"):
        (rec_dir / sub / "notes.json").write_text(
            json.dumps({"title": f"{sub} mtg", "date": "2026-07-10"}), encoding="utf-8")
    monkeypatch.setenv("BRAIN_JUSTREC_DIR", str(rec_dir))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["justrec"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 2   # a/notes.json and b/notes.json are distinct ids


def test_justrec_no_dir_pulls_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BRAIN_JUSTREC_DIR", raising=False)
    stats = run_connector(CONNECTORS["justrec"](), _vault(tmp_path), pulled_at=_AT, logger=_LOG)
    assert stats.written == 0
