"""Core ingest-pipeline tests over throwaway tmp-path vaults.

Covers the repo's headline invariants — none of which had a test before:
SHA-256 idempotency skip, re-process on content change, archive-immutability
hash-clash -> manual_review, failed-file de-duplication (identical bytes
reused, different bytes suffixed), a manual_review-only run still refreshing
the status dashboards, the backfill Processing-notes de-duplication, and
metadata unknown-key + torn-tail tolerance.

Summarization is disabled (BRAIN_SKIP_SUMMARY) so nothing hits an LLM.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ingest_lib.config import paths_for_root  # type: ignore[import-not-found]
from ingest_lib.metadata import (  # type: ignore[import-not-found]
    IndexRecord,
    append_record,
    iter_records,
    latest_records_by_path,
)
from ingest_lib.pipeline import (  # type: ignore[import-not-found]
    _move_to_failed,
    _strip_frontmatter_header,
    plan_ingest,
    run_ingest,
)

_LOG = logging.getLogger("test")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("BRAIN_SKIP_SUMMARY", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _vault(tmp_path: Path):
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def _drop(paths, rel: str, text: str) -> Path:
    p = paths.inbox / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _ingest(paths):
    plan = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG)
    return run_ingest(paths, plan, dry_run=False, logger=_LOG)


# --------------------------------------------------------- idempotency

def test_reingest_same_content_is_a_noop(tmp_path: Path):
    paths = _vault(tmp_path)
    _drop(paths, "notes/a.txt", "hello world\n")
    s1 = _ingest(paths)
    assert s1.processed == 1
    lines_after_first = paths.metadata_index_jsonl.read_text().count("\n")

    s2 = _ingest(paths)
    # Second run skips the unchanged file — no new JSONL record.
    assert s2.processed == 0
    assert s2.skipped == 1
    assert paths.metadata_index_jsonl.read_text().count("\n") == lines_after_first


def test_changed_content_is_reprocessed_from_archive(tmp_path: Path):
    # Reprocess-on-changed-hash is the --raw path: re-running extraction over
    # archive/raw. (From inbox, a changed file clashing with an existing raw
    # twin is a manual_review, since raw is immutable — see the clash test.)
    paths = _vault(tmp_path)
    raw = paths.archive_raw / "notes/a.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("version one\n", encoding="utf-8")

    plan1 = plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    assert run_ingest(paths, plan1, dry_run=False, logger=_LOG).processed == 1

    raw.write_text("version two — totally different\n", encoding="utf-8")
    plan2 = plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    assert run_ingest(paths, plan2, dry_run=False, logger=_LOG).processed == 1

    recs = [r for r in iter_records(paths.metadata_index_jsonl) if r.relative_path == "notes/a.txt"]
    assert len(recs) == 2  # a fresh record appended, old one retained


# ------------------------------------------ archive immutability / clash

def test_same_stem_different_ext_do_not_clobber(tmp_path: Path):
    # D7: report.pdf and report.docx in one folder must produce distinct
    # processed/index notes and assets dirs (both kept via the extension),
    # not both collapse onto report.md.
    paths = _vault(tmp_path)
    _drop(paths, "notes/report.txt", "the txt report body\n")
    _drop(paths, "notes/report.md", "# the markdown report body\n")
    stats = _ingest(paths)
    assert stats.processed == 2
    recs = {r.relative_path: r for r in latest_records_by_path(paths.metadata_index_jsonl).values()}
    # Distinct processed paths, both carrying the source extension.
    assert recs["notes/report.txt"].processed_path == "archive/processed/notes/report.txt.md"
    assert recs["notes/report.md"].processed_path == "archive/processed/notes/report.md.md"
    assert (paths.root / "archive/processed/notes/report.txt.md").is_file()
    assert (paths.root / "archive/processed/notes/report.md.md").is_file()


def test_inbox_hash_clash_with_raw_is_manual_review(tmp_path: Path):
    paths = _vault(tmp_path)
    # A raw file already exists at this path with DIFFERENT content.
    raw = paths.archive_raw / "notes/a.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("the immutable original\n", encoding="utf-8")
    raw_bytes_before = raw.read_bytes()

    _drop(paths, "notes/a.txt", "an incoming DIFFERENT version\n")
    stats = _ingest(paths)

    assert stats.manual_review == 1
    # Raw ground truth is untouched, byte-for-byte.
    assert raw.read_bytes() == raw_bytes_before
    latest = latest_records_by_path(paths.metadata_index_jsonl)["notes/a.txt"]
    assert latest.status == "manual_review"
    assert latest.extractor == "archive-clash"


# ---------------------------------------- _strip_frontmatter_header (F049)

_PROCESSED = (
    "# A Title\n\n"
    "> Source: `notes/a.txt`  \n"
    "> Hash: `abc`  \n"
    "> Extractor: `text`  \n"
    "> Status: `processed`\n\n"
    "---\n\n"
    "The real extracted body.\n\n"
    "Second paragraph.\n\n"
    "---\n\n"
    "## Processing notes\n\n"
    "- Extractor: `text`\n"
)


def test_strip_header_drops_leading_and_trailing_blocks():
    body = _strip_frontmatter_header(_PROCESSED)
    assert body.strip() == "The real extracted body.\n\nSecond paragraph."
    assert "## Processing notes" not in body
    assert "Source:" not in body


def test_strip_header_heals_already_duplicated_footer():
    # A file corrupted by the old backfill has TWO footers; stripping must
    # remove both so a re-backfill produces a single Processing-notes block.
    doubled = _PROCESSED + "\n---\n\n## Processing notes\n\n- summary: reused\n"
    body = _strip_frontmatter_header(doubled)
    assert body.count("## Processing notes") == 0
    assert body.strip() == "The real extracted body.\n\nSecond paragraph."


# --------------------------------------------- metadata unknown keys (F041)

def test_iter_records_tolerates_unknown_and_missing_keys(tmp_path: Path):
    paths = _vault(tmp_path)
    rec = IndexRecord(
        relative_path="notes/a.txt", source_hash="h", size_bytes=1,
        extension=".txt", extractor="text", status="processed",
        raw_path="archive/raw/notes/a.txt", processed_path=None,
        index_note_path=None,
    )
    append_record(paths.metadata_index_jsonl, rec)
    # A line from a newer tool version with an extra field, plus a garbage line.
    with paths.metadata_index_jsonl.open("a", encoding="utf-8") as fh:
        fh.write('{"relative_path":"notes/b.txt","source_hash":"h2","size_bytes":2,'
                 '"extension":".txt","extractor":"text","status":"processed",'
                 '"raw_path":"x","processed_path":null,"index_note_path":null,'
                 '"future_field":"ignore me"}\n')
        fh.write("{ this is not json }\n")
    got = {r.relative_path for r in iter_records(paths.metadata_index_jsonl)}
    assert got == {"notes/a.txt", "notes/b.txt"}  # unknown key dropped, garbage skipped


def test_append_record_self_heals_a_torn_tail(tmp_path: Path):
    paths = _vault(tmp_path)
    # Simulate a torn prior write: a partial line with no trailing newline.
    paths.metadata_index_jsonl.write_text('{"partial": "no newline"', encoding="utf-8")
    rec = IndexRecord(
        relative_path="notes/a.txt", source_hash="h", size_bytes=1,
        extension=".txt", extractor="text", status="processed",
        raw_path="x", processed_path=None, index_note_path=None,
    )
    append_record(paths.metadata_index_jsonl, rec)
    # The new record lands on its own line — recoverable, not merged into the stub.
    recs = list(iter_records(paths.metadata_index_jsonl))
    assert [r.relative_path for r in recs] == ["notes/a.txt"]


@pytest.mark.parametrize("tail", [
    '{"s": "Wuczyń'.encode(),          # torn on the final byte of 'ń'
    '{"s": "Wuczyń'.encode()[:-1],     # torn mid multibyte sequence
])
def test_append_record_self_heals_a_multibyte_torn_tail(tmp_path: Path, tail: bytes):
    # The torn-tail probe must not decode the last byte as text: a tail ending
    # mid-UTF-8 raised UnicodeDecodeError before any write, crashing every retry.
    paths = _vault(tmp_path)
    paths.metadata_index_jsonl.write_bytes(tail)
    rec = IndexRecord(
        relative_path="notes/a.txt", source_hash="h", size_bytes=1,
        extension=".txt", extractor="text", status="processed",
        raw_path="x", processed_path=None, index_note_path=None,
    )
    append_record(paths.metadata_index_jsonl, rec)
    recs = list(iter_records(paths.metadata_index_jsonl))
    assert [r.relative_path for r in recs] == ["notes/a.txt"]


# --------------------------------------------- failed-file de-duplication (F2)

def test_failed_move_dedupes_identical_bytes(tmp_path: Path):
    # A file that keeps failing on every re-run must not stack byte-identical
    # copies (x.docx, x.docx.1, x.docx.2, ...) in archive/failed/.
    paths = _vault(tmp_path)
    rel = "bad.docx"
    (paths.archive_raw).mkdir(parents=True, exist_ok=True)

    def _raw_with(content: bytes) -> Path:
        p = paths.archive_raw / rel
        p.write_bytes(content)
        return p

    first = _move_to_failed(_raw_with(b"corrupt"), paths)
    assert first is not None and first.name == "bad.docx"
    # Same bytes fail again: reuse the existing copy, don't mint '.1'.
    second = _move_to_failed(_raw_with(b"corrupt"), paths)
    assert second == first
    assert sorted(p.name for p in paths.archive_failed.glob("bad.docx*")) == ["bad.docx"]
    # Different bytes DO get a distinct suffix (no data loss).
    third = _move_to_failed(_raw_with(b"a different corruption"), paths)
    assert third is not None and third.name == "bad.docx.1"


def test_manual_review_only_run_refreshes_status_dashboard(tmp_path: Path):
    # A run that only produces failures must still write the Manual Review
    # dashboard — that's the run whose whole point is surfacing the failure.
    paths = _vault(tmp_path)
    p = paths.inbox / "bad.docx"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a real docx zip")  # docx extractor -> manual_review

    stats = _ingest(paths)
    assert stats.manual_review == 1
    assert stats.processed == 0 and stats.partial == 0

    review = paths.knowledge_index / "Manual Review.md"
    assert review.exists()
    assert "bad.docx" in review.read_text(encoding="utf-8")


def test_failed_reextraction_preserves_previous_assets(tmp_path: Path, monkeypatch):
    # F5: a re-extraction that FAILS must not destroy the previous good
    # extraction's assets (archive/processed is regenerable, but only from a
    # SUCCESSFUL run — a failed one leaves nothing to regenerate from).
    from ingest_lib import pipeline as pl
    from ingest_lib.extractors import ExtractionResult

    paths = _vault(tmp_path)
    rel = "notes/doc.pdf"
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"version one")

    calls = {"n": 0}

    def fake_dispatch(_src, relative_path=None):
        def extract(src: Path, assets_dir: Path) -> ExtractionResult:
            calls["n"] += 1
            if calls["n"] == 1:
                assets_dir.mkdir(parents=True, exist_ok=True)
                (assets_dir / "fig1.png").write_bytes(b"figure-bytes")
                return ExtractionResult(status="processed", extractor="stub",
                                        markdown="body one\n",
                                        assets=[assets_dir / "fig1.png"])
            raise RuntimeError("extractor blew up on re-run")
        return extract

    monkeypatch.setattr(pl, "dispatch_extractor", fake_dispatch)

    plan1 = pl.plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    assert pl.run_ingest(paths, plan1, dry_run=False, logger=_LOG).processed == 1
    asset = paths.root / "archive/processed/notes/doc.pdf_assets/fig1.png"
    assert asset.is_file()

    # Change the raw bytes so it re-extracts; the extractor now fails.
    raw.write_bytes(b"version two - different")
    plan2 = pl.plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    pl.run_ingest(paths, plan2, dry_run=False, logger=_LOG)

    # The previous good asset must still exist (failed run preserved it).
    assert asset.is_file()
    assert asset.read_bytes() == b"figure-bytes"
