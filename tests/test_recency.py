"""Tests for ingest_lib.recency — half-life decay math and the re-ranked
memory_search. semantic.search is monkeypatched to fixed SearchHits so no
model or index files are ever touched."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest

from ingest_lib import semantic
from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.metadata import IndexRecord, append_record
from ingest_lib.recency import MemoryHit, memory_search, recency_weight
from ingest_lib.semantic import SearchHit

_LOG = logging.getLogger("test")
NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# recency_weight
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_halflife_halves_the_weight() -> None:
    one_halflife_ago = _iso(NOW - timedelta(days=30))
    assert recency_weight(one_halflife_ago, halflife_days=30.0, now=NOW) == pytest.approx(0.5)
    two_halflives_ago = _iso(NOW - timedelta(days=60))
    assert recency_weight(two_halflives_ago, halflife_days=30.0, now=NOW) == pytest.approx(0.25)


def test_empty_and_garbage_dates_are_timeless() -> None:
    # Timeless notes are not penalised: no clock means weight 1.0.
    assert recency_weight("", halflife_days=30.0, now=NOW) == 1.0
    assert recency_weight("   ", halflife_days=30.0, now=NOW) == 1.0
    assert recency_weight("not-a-date", halflife_days=30.0, now=NOW) == 1.0


def test_date_only_strings_parse() -> None:
    # Bare YYYY-MM-DD parses to midnight UTC: 12h old at NOW.
    w = recency_weight("2026-06-12", halflife_days=30.0, now=NOW)
    assert w == pytest.approx(0.5 ** (0.5 / 30.0))


def test_future_dates_weigh_one() -> None:
    assert recency_weight("2027-01-01", halflife_days=30.0, now=NOW) == 1.0
    assert recency_weight(_iso(NOW), halflife_days=30.0, now=NOW) == 1.0


def test_memory_search_rejects_nonpositive_halflife(tmp_path: Path) -> None:
    # memory_search is public library API; halflife_days<=0 otherwise reaches
    # recency_weight and raises ZeroDivisionError / OverflowError deep inside.
    # Validate it at the boundary like the `types` parameter already is.
    paths = paths_for_root(tmp_path)
    paths.ensure()
    for bad in (0, -30):
        with pytest.raises(ValueError, match="halflife_days"):
            memory_search(paths, "q", halflife_days=bad, now=NOW, logger=_LOG)


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------

def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _note(paths: VaultPaths, rel: str, *, updated: str = "", status: str = "") -> None:
    fm = ["---", 'title: "x"']
    if updated:
        fm.append(f'updated: "{updated}"')
    if status:
        fm.append(f"memory_status: {status}")
    fm.append("---")
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(fm) + "\n\nBody paragraph long enough to matter.\n", encoding="utf-8")


def _hit(rel: str, score: float = 0.9, origin: str = "knowledge-note") -> SearchHit:
    return SearchHit(
        score=score, source_relative_path=rel, title=Path(rel).stem,
        chunk_idx=0, snippet="snippet", origin=origin,
    )


def _patch_search(monkeypatch: pytest.MonkeyPatch, hits: list[SearchHit]) -> list[int]:
    requested: list[int] = []

    def fake_search(paths: VaultPaths, query: str, *, top_k: int = 10, logger=None):
        requested.append(top_k)
        return hits

    monkeypatch.setattr(semantic, "search", fake_search)
    return requested


def test_fresher_note_outranks_stale_at_equal_cosine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _note(paths, "knowledge/notes/fresh.md", updated=_iso(NOW - timedelta(days=1)))
    _note(paths, "knowledge/notes/stale.md", updated=_iso(NOW - timedelta(days=120)))
    requested = _patch_search(
        monkeypatch,
        [_hit("knowledge/notes/stale.md"), _hit("knowledge/notes/fresh.md")],
    )

    hits = memory_search(paths, "q", top_k=5, now=NOW, logger=_LOG)

    assert [h.source_relative_path for h in hits] == [
        "knowledge/notes/fresh.md", "knowledge/notes/stale.md",
    ]
    assert hits[0].cosine == hits[1].cosine == pytest.approx(0.9)
    assert hits[0].recency > hits[1].recency
    assert hits[0].score > hits[1].score
    # Candidate over-fetch: floor of 50 even for small top_k.
    assert requested == [50]


def test_superseded_note_sinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _vault(tmp_path)
    when = _iso(NOW - timedelta(days=2))
    _note(paths, "knowledge/people/old.md", updated=when, status="superseded")
    _note(paths, "knowledge/people/new.md", updated=when, status="unconsolidated")
    _patch_search(
        monkeypatch,
        [_hit("knowledge/people/old.md"), _hit("knowledge/people/new.md")],
    )

    hits = memory_search(paths, "q", now=NOW, logger=_LOG)

    assert [h.source_relative_path for h in hits] == [
        "knowledge/people/new.md", "knowledge/people/old.md",
    ]
    assert hits[1].status_weight == pytest.approx(0.2)
    assert hits[0].status_weight == 1.0  # unconsolidated is not demoted
    assert hits[1].score == pytest.approx(hits[1].cosine * hits[1].recency * 0.2)


def test_types_filter_includes_and_excludes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _note(paths, "knowledge/people/anna.md", updated=_iso(NOW))
    _note(paths, "knowledge/projects/kern/kern.md", updated=_iso(NOW))
    all_hits = [
        _hit("knowledge/people/anna.md"),
        _hit("knowledge/projects/kern/kern.md"),
        _hit("uni/lecture.pdf", origin="pdf-mineru"),
    ]
    _patch_search(monkeypatch, all_hits)

    people = memory_search(paths, "q", types=["people"], now=NOW, logger=_LOG)
    assert [h.source_relative_path for h in people] == ["knowledge/people/anna.md"]

    archive = memory_search(paths, "q", types=["archive"], now=NOW, logger=_LOG)
    assert [h.source_relative_path for h in archive] == ["uni/lecture.pdf"]

    both = memory_search(paths, "q", types=["projects", "archive"], now=NOW, logger=_LOG)
    assert {h.source_relative_path for h in both} == {
        "knowledge/projects/kern/kern.md", "uni/lecture.pdf",
    }

    everything = memory_search(paths, "q", types=None, now=NOW, logger=_LOG)
    assert len(everything) == 3


def test_types_filter_overfetches_so_sparse_type_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A few people notes ranked DEEP below a flood of archive hits. With a
    # small candidate pool the filter would starve them out entirely.
    paths = _vault(tmp_path)
    _note(paths, "knowledge/people/anna.md", updated=_iso(NOW))
    _note(paths, "knowledge/people/bob.md", updated=_iso(NOW))
    archive_hits = [
        _hit(f"uni/lecture-{i}.pdf", score=0.9, origin="pdf-mineru")
        for i in range(200)
    ]
    people_hits = [
        _hit("knowledge/people/anna.md", score=0.1),
        _hit("knowledge/people/bob.md", score=0.1),
    ]
    requested = _patch_search(monkeypatch, archive_hits + people_hits)

    people = memory_search(paths, "q", types=["people"], top_k=10, now=NOW, logger=_LOG)

    assert {h.source_relative_path for h in people} == {
        "knowledge/people/anna.md", "knowledge/people/bob.md",
    }
    # A filtered query over-fetches a large candidate pool (cap 500) so deep
    # sparse-type matches still survive the post-fetch filter.
    assert requested == [500]


def test_filter_classifies_by_origin_not_path_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An ingested PDF dropped at inbox/knowledge/people/lecture.pdf keeps a
    # "knowledge/people/" label but is NOT a vault note — origin (the PDF
    # extractor, not KNOWLEDGE_EXTRACTOR) is what disambiguates it.
    paths = _vault(tmp_path)
    _note(paths, "knowledge/people/anna.md", updated=_iso(NOW))
    _patch_search(
        monkeypatch,
        [
            _hit("knowledge/people/lecture.pdf", origin="pdf-mineru"),
            _hit("knowledge/people/anna.md", origin="knowledge-note"),
        ],
    )

    # types=["people"] selects the real note, not the ingested PDF.
    people = memory_search(paths, "q", types=["people"], now=NOW, logger=_LOG)
    assert [h.source_relative_path for h in people] == ["knowledge/people/anna.md"]

    # types=["archive"] selects the ingested PDF, not the real note.
    archive = memory_search(paths, "q", types=["archive"], now=NOW, logger=_LOG)
    assert [h.source_relative_path for h in archive] == ["knowledge/people/lecture.pdf"]


def test_unknown_type_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _vault(tmp_path)
    _patch_search(monkeypatch, [])
    with pytest.raises(ValueError, match="bogus"):
        memory_search(paths, "q", types=["bogus"], now=NOW, logger=_LOG)
    # Validation lists the valid tokens so the error is actionable.
    with pytest.raises(ValueError, match="archive"):
        memory_search(paths, "q", types=["people", "nope"], now=NOW, logger=_LOG)


def test_archive_hits_join_updated_at_from_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    append_record(
        paths.metadata_index_jsonl,
        IndexRecord(
            relative_path="uni/lecture.pdf", source_hash="h", size_bytes=1,
            extension=".pdf", extractor="pdf-mineru", status="processed",
            raw_path="archive/raw/uni/lecture.pdf",
            processed_path="archive/processed/uni/lecture.md",
            index_note_path=None, updated_at=_iso(NOW - timedelta(days=30)),
        ),
    )
    _patch_search(monkeypatch, [_hit("uni/lecture.pdf", origin="pdf-mineru")])

    hits = memory_search(paths, "q", halflife_days=30.0, now=NOW, logger=_LOG)

    assert len(hits) == 1
    assert hits[0].updated == _iso(NOW - timedelta(days=30))
    assert hits[0].recency == pytest.approx(0.5)
    assert hits[0].status_weight == 1.0  # ingested sources have no memory_status


def test_missing_note_file_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    # Hit references a knowledge note that no longer exists on disk.
    _patch_search(monkeypatch, [_hit("knowledge/notes/ghost.md")])

    hits = memory_search(paths, "q", now=NOW, logger=_LOG)

    assert len(hits) == 1
    assert hits[0].updated == ""
    assert hits[0].recency == 1.0  # timeless, not an error


def test_top_k_truncates_after_reranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _note(paths, "knowledge/notes/fresh.md", updated=_iso(NOW))
    _note(paths, "knowledge/notes/stale.md", updated=_iso(NOW - timedelta(days=365)))
    # Stale wins on cosine but loses after decay; top_k=1 keeps fresh only.
    _patch_search(
        monkeypatch,
        [_hit("knowledge/notes/stale.md", score=0.95), _hit("knowledge/notes/fresh.md", score=0.9)],
    )

    hits = memory_search(paths, "q", top_k=1, now=NOW, logger=_LOG)

    assert len(hits) == 1
    assert hits[0].source_relative_path == "knowledge/notes/fresh.md"
    assert isinstance(hits[0], MemoryHit)
