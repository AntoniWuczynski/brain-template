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
import re
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.metadata import (
    IndexRecord,
    append_record,
    iter_records,
    latest_records_by_path,
)
from ingest_lib.notes import derived_assets_dirname, sanitise_derived_name
from ingest_lib.pipeline import (
    IngestStats,
    _move_to_failed,
    _strip_frontmatter_header,
    plan_ingest,
    run_ingest,
)

_LOG = logging.getLogger("test")
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def _wikilinks(text: str) -> list[str]:
    """Link targets the way a wikilink parser sees them — whitespace inside
    the brackets stripped (Obsidian and scripts/sweep.py both do this)."""
    return [m.strip() for m in _WIKILINK.findall(text)]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_SKIP_SUMMARY", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def _drop(paths: VaultPaths, rel: str, text: str) -> Path:
    p = paths.inbox / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _ingest(paths: VaultPaths) -> IngestStats:
    plan = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG)
    return run_ingest(paths, plan, dry_run=False, logger=_LOG)


# --------------------------------------------------------- idempotency

def test_reingest_same_content_is_a_noop(tmp_path: Path) -> None:
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


def test_changed_content_is_reprocessed_from_archive(tmp_path: Path) -> None:
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

def test_same_stem_different_ext_do_not_clobber(tmp_path: Path) -> None:
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


def test_inbox_hash_clash_with_raw_is_manual_review(tmp_path: Path) -> None:
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


def test_strip_header_drops_leading_and_trailing_blocks() -> None:
    body = _strip_frontmatter_header(_PROCESSED)
    assert body.strip() == "The real extracted body.\n\nSecond paragraph."
    assert "## Processing notes" not in body
    assert "Source:" not in body


def test_strip_header_heals_already_duplicated_footer() -> None:
    # A file corrupted by the old backfill has TWO footers; stripping must
    # remove both so a re-backfill produces a single Processing-notes block.
    doubled = _PROCESSED + "\n---\n\n## Processing notes\n\n- summary: reused\n"
    body = _strip_frontmatter_header(doubled)
    assert body.count("## Processing notes") == 0
    assert body.strip() == "The real extracted body.\n\nSecond paragraph."


# --------------------------------------------- metadata unknown keys (F041)

def test_iter_records_tolerates_unknown_and_missing_keys(tmp_path: Path) -> None:
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


def test_append_record_self_heals_a_torn_tail(tmp_path: Path) -> None:
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
def test_append_record_self_heals_a_multibyte_torn_tail(tmp_path: Path, tail: bytes) -> None:
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

def test_failed_move_dedupes_identical_bytes(tmp_path: Path) -> None:
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


def test_manual_review_only_run_refreshes_status_dashboard(tmp_path: Path) -> None:
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


def test_failed_reextraction_preserves_previous_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # F5: a re-extraction that FAILS must not destroy the previous good
    # extraction's assets (archive/processed is regenerable, but only from a
    # SUCCESSFUL run — a failed one leaves nothing to regenerate from).
    from ingest_lib import pipeline as pl
    from ingest_lib.extractors import Extractor, ExtractionResult

    paths = _vault(tmp_path)
    rel = "notes/doc.pdf"
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"version one")

    calls = {"n": 0}

    def fake_dispatch(_src: Path, relative_path: str | None = None) -> Extractor:
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


def test_index_note_lists_extracted_figures(tmp_path: Path) -> None:
    # 1.a: image assets an extractor produces are listed in the index note's
    # `figures:` frontmatter for fast visual review in Obsidian.
    from ingest_lib.notes import NoteContent, write_index_note

    target = tmp_path / "knowledge/index/uni/lec.pdf.md"
    write_index_note(target=target, content=NoteContent(
        title="Lec", source_relative_path="uni/lec.pdf", source_hash="h",
        status="processed", extracted_markdown="body\n", processing_notes=[],
        extractor="pdf-mineru",
        figures=("archive/processed/uni/lec.pdf_assets/fig1.jpg",
                 "archive/processed/uni/lec.pdf_assets/fig2.png"),
    ))
    text = target.read_text(encoding="utf-8")
    assert "figures:" in text
    assert "archive/processed/uni/lec.pdf_assets/fig1.jpg" in text
    assert "fig2.png" in text


def test_pipeline_records_image_figures_in_index_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib import pipeline as pl
    from ingest_lib.extractors import Extractor, ExtractionResult

    paths = _vault(tmp_path)
    raw = paths.archive_raw / "uni/doc.pdf"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"pdf-bytes")

    def fake_dispatch(_src: Path, relative_path: str | None = None) -> Extractor:
        def extract(src: Path, assets_dir: Path) -> ExtractionResult:
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / "fig1.png").write_bytes(b"img")
            (assets_dir / "table.html").write_text("<table></table>", encoding="utf-8")
            return ExtractionResult(status="processed", extractor="stub",
                                    markdown="body\n",
                                    assets=[assets_dir / "fig1.png", assets_dir / "table.html"])
        return extract

    monkeypatch.setattr(pl, "dispatch_extractor", fake_dispatch)
    plan = pl.plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    pl.run_ingest(paths, plan, dry_run=False, logger=_LOG)

    note = (paths.knowledge_index / "uni/doc.pdf.md").read_text(encoding="utf-8")
    # Exact vault-relative path (not a bare basename that an absolute or
    # pre-swap temp path would also satisfy).
    assert "- archive/processed/uni/doc.pdf_assets/fig1.png" in note
    assert "table.html" not in note           # non-image asset not a "figure"


def test_figures_refresh_and_drop_on_reingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Managed key: figures refresh when they change, and the key is dropped
    # entirely when a re-extraction yields none (no stale frontmatter).
    from ingest_lib import pipeline as pl
    from ingest_lib.extractors import Extractor, ExtractionResult

    paths = _vault(tmp_path)
    raw = paths.archive_raw / "uni/doc.pdf"
    raw.parent.mkdir(parents=True, exist_ok=True)
    assets = {"n": 0}

    def fake_dispatch(_src: Path, relative_path: str | None = None) -> Extractor:
        def extract(src: Path, assets_dir: Path) -> ExtractionResult:
            assets_dir.mkdir(parents=True, exist_ok=True)
            names = ([ "fig1.png" ], [ "figA.jpg" ], [])[assets["n"]]
            for nm in names:
                (assets_dir / nm).write_bytes(b"img")
            return ExtractionResult(status="processed", extractor="stub",
                                    markdown="body\n",
                                    assets=[assets_dir / nm for nm in names])
        return extract

    monkeypatch.setattr(pl, "dispatch_extractor", fake_dispatch)

    def ingest(bytes_: bytes) -> str:
        raw.write_bytes(bytes_)
        plan = pl.plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
        pl.run_ingest(paths, plan, dry_run=False, logger=_LOG)
        return (paths.knowledge_index / "uni/doc.pdf.md").read_text(encoding="utf-8")

    assets["n"] = 0
    n1 = ingest(b"v1")
    assert "fig1.png" in n1
    assets["n"] = 1
    n2 = ingest(b"v2")
    assert "figA.jpg" in n2 and "fig1.png" not in n2   # refreshed
    assets["n"] = 2
    n3 = ingest(b"v3")
    assert "figures:" not in n3                          # dropped, no stale key


# ------------------------------------------- raw-copy atomicity (F3)

def test_torn_raw_copy_leaves_no_file_in_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # F3: a crash mid-copy must not leave a truncated file under archive/raw
    # (which the next run would read as a permanent archive-clash). The copy
    # lands on a temp name and is renamed into place.
    import shutil

    paths = _vault(tmp_path)
    src = _drop(paths, "notes/big.txt", "the full body\n" * 20)

    def torn_copy2(s: str | Path, d: str | Path, *, follow_symlinks: bool = True) -> str:
        Path(d).write_bytes(Path(s).read_bytes()[:5])
        raise OSError("disk full mid-copy")

    monkeypatch.setattr(shutil, "copy2", torn_copy2)
    stats = _ingest(paths)

    raw = paths.archive_raw / "notes/big.txt"
    assert not raw.exists()                       # nothing torn in the immutable tree
    assert list(raw.parent.glob("*")) == []       # and no temp left behind
    assert stats.manual_review == 1
    latest = latest_records_by_path(paths.metadata_index_jsonl)["notes/big.txt"]
    assert latest.status == "manual_review"

    # The retry succeeds: no phantom clash blocks it.
    monkeypatch.undo()
    assert _ingest(paths).processed == 1
    assert raw.read_bytes() == src.read_bytes()


# --------------------------------------- per-file crash isolation (F4)

def test_one_crashing_file_does_not_abort_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # F4: an exception raised while writing one file's notes (e.g. a
    # hand-edited index note with a non-string frontmatter key) must be
    # recorded as manual_review, not abort every later file in the plan.
    from ingest_lib.notes import NoteContent, write_index_note

    def boom(*, target: Path, content: NoteContent) -> None:
        if target.name.startswith("a.txt"):
            raise TypeError("'<' not supported between instances of 'bool' and 'str'")
        write_index_note(target=target, content=content)

    monkeypatch.setattr("ingest_lib.pipeline.write_index_note", boom)

    paths = _vault(tmp_path)
    _drop(paths, "notes/a.txt", "first\n")
    _drop(paths, "notes/b.txt", "second\n")
    stats = _ingest(paths)

    assert stats.manual_review == 1 and stats.processed == 1
    latest = latest_records_by_path(paths.metadata_index_jsonl)
    assert latest["notes/a.txt"].status == "manual_review"
    assert "TypeError" in (latest["notes/a.txt"].error or "")
    assert latest["notes/b.txt"].status == "processed"


# ------------------------------------ unicode path normalisation (F7)

def test_nfd_filename_matches_an_nfc_index_record(tmp_path: Path) -> None:
    # F7: macOS hands back NFD names while git/Obsidian write NFC; an
    # already-processed file must not be re-planned because of the spelling.
    import unicodedata

    from ingest_lib.hashing import sha256_of

    paths = _vault(tmp_path)
    nfd = unicodedata.normalize("NFD", "notes/café.txt")
    nfc = unicodedata.normalize("NFC", "notes/café.txt")
    src = _drop(paths, nfd, "body\n")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=nfc,
        source_hash=sha256_of(src),
        size_bytes=src.stat().st_size,
        extension=".txt",
        extractor="text",
        status="processed",
        raw_path="archive/raw/" + nfc,
        processed_path="archive/processed/" + nfc + ".md",
        index_note_path="knowledge/index/" + nfc + ".md",
    ))

    plan = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG)
    assert [i.relative_path for i in plan.items] == []
    assert [i.relative_path for i in plan.skipped_already_processed] == [nfc]


# ------------------------------- failed re-extraction keeps raw (F10)

def test_failed_reextraction_leaves_raw_in_archive(tmp_path: Path) -> None:
    # F10: archive/raw is immutable (AGENTS.md hard rule 1) — a re-extraction
    # that fails must mark the record and leave ground truth where it is.
    paths = _vault(tmp_path)
    raw = paths.archive_raw / "notes/bad.docx"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"not a real docx zip")

    plan = plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    stats = run_ingest(paths, plan, dry_run=False, logger=_LOG)

    assert stats.manual_review == 1
    assert raw.read_bytes() == b"not a real docx zip"
    assert list(paths.archive_failed.rglob("*.docx")) == []
    latest = latest_records_by_path(paths.metadata_index_jsonl)["notes/bad.docx"]
    assert latest.status == "manual_review"
    assert latest.raw_path == "archive/raw/notes/bad.docx"


def test_first_time_inbox_failure_still_moves_raw_to_failed(tmp_path: Path) -> None:
    # The other side of F10: a file that failed on its FIRST ingest never had
    # a good archived copy, so rule 6's move to archive/failed still applies.
    paths = _vault(tmp_path)
    _drop(paths, "notes/bad.docx", "not a real docx zip")
    stats = _ingest(paths)

    assert stats.manual_review == 1
    assert not (paths.archive_raw / "notes/bad.docx").exists()
    assert (paths.archive_failed / "notes/bad.docx").is_file()
    latest = latest_records_by_path(paths.metadata_index_jsonl)["notes/bad.docx"]
    assert latest.raw_path == "archive/failed/notes/bad.docx"


# ----------------------------------- honest archive-clash record (F12)

def test_archive_clash_record_describes_its_raw_path(tmp_path: Path) -> None:
    # F12: the clash record pointed at the archived file while its hash/size
    # described the incoming one. Hash and path must describe one file.
    from ingest_lib.hashing import sha256_of

    paths = _vault(tmp_path)
    raw = paths.archive_raw / "notes/a.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("the immutable original\n", encoding="utf-8")
    incoming = _drop(paths, "notes/a.txt", "an incoming DIFFERENT version\n")

    _ingest(paths)
    latest = latest_records_by_path(paths.metadata_index_jsonl)["notes/a.txt"]
    assert latest.extractor == "archive-clash"
    assert latest.raw_path == "archive/raw/notes/a.txt"
    assert latest.source_hash == sha256_of(raw)
    assert latest.size_bytes == raw.stat().st_size
    # The incoming bytes are still described — in the error, not the record's
    # identity fields.
    assert sha256_of(incoming)[:12] in (latest.error or "")


def test_leftover_raw_copy_temp_is_not_a_source(tmp_path: Path) -> None:
    # The temp name _atomic_copy writes to is not a source: a kill mid-copy
    # must not get its truncated bytes ingested on the next run.
    paths = _vault(tmp_path)
    (paths.archive_raw / "notes").mkdir(parents=True, exist_ok=True)
    (paths.archive_raw / "notes/.tmp-big.txt").write_text("trunc", encoding="utf-8")

    plan = plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    assert plan.items == [] and plan.skipped_unsupported == []


# ------------------------------- archive clash is recorded once (AUD-016)

def test_archive_clash_is_recorded_once_not_every_run(tmp_path: Path) -> None:
    # AUD-016: an edited inbox file clashes with its immutable archived copy.
    # The index is append-only, so re-recording the identical failure on every
    # run just grows the JSONL — one record, however many runs.
    paths = _vault(tmp_path)
    _drop(paths, "note.txt", "version one\n")
    _ingest(paths)
    _drop(paths, "note.txt", "version two — edited in place\n")
    first = _ingest(paths)
    second = _ingest(paths)

    assert first.manual_review == 1 and second.manual_review == 1
    clashes = [r for r in iter_records(paths.metadata_index_jsonl)
               if r.extractor == "archive-clash"]
    assert len(clashes) == 1


# ------------------------ manual_review gets an index note (AUD-017)

def test_manual_review_writes_an_index_note(tmp_path: Path) -> None:
    # AUD-017: a failed source had no note at all, so `status: manual_review`
    # never appeared in knowledge/index/ and the error lived only in the JSONL.
    paths = _vault(tmp_path)
    _drop(paths, "notes/bad.docx", "not a real docx zip")
    _ingest(paths)

    note = paths.knowledge_index / "notes/bad.docx.md"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "status: manual_review" in text
    assert "_(empty — extraction failed; see Processing notes)_" in text
    assert "archive/failed/notes/bad.docx" in text
    rec = latest_records_by_path(paths.metadata_index_jsonl)["notes/bad.docx"]
    assert rec.index_note_path == "knowledge/index/notes/bad.docx.md"


def test_failed_reextraction_keeps_the_previous_good_note(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    # The other side of AUD-017: the failure note must never blank a note a
    # successful extraction already wrote (same rule as the preserved assets).
    from ingest_lib import pipeline as pl
    from ingest_lib.extractors import Extractor, ExtractionResult

    paths = _vault(tmp_path)
    raw = paths.archive_raw / "notes/doc.docx"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("one", encoding="utf-8")
    calls = {"n": 0}

    def fake_dispatch(_src: Path, relative_path: str | None = None) -> Extractor:
        def extract(src: Path, assets_dir: Path) -> ExtractionResult:
            calls["n"] += 1
            if calls["n"] == 1:
                return ExtractionResult(status="processed", extractor="stub",
                                        markdown="good body\n")
            return ExtractionResult(status="manual_review", extractor="stub",
                                    markdown="", error="second pass failed")
        return extract

    monkeypatch.setattr(pl, "dispatch_extractor", fake_dispatch)
    plan = pl.plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    pl.run_ingest(paths, plan, dry_run=False, logger=_LOG)
    raw.write_text("two", encoding="utf-8")
    plan = pl.plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    pl.run_ingest(paths, plan, dry_run=False, logger=_LOG)

    note = paths.knowledge_index / "notes/doc.docx.md"
    assert "status: processed" in note.read_text(encoding="utf-8")


# ------------------------- legacy derived-note names (AUD-018)

def test_reingest_migrates_a_legacy_scheme_note(tmp_path: Path) -> None:
    # AUD-018: 624 live records predate the extension-keeping derived name
    # (report.pdf -> report.pdf.md). A re-ingest used to write the new twin
    # beside the old note, orphaning the user's edits and the assets dir.
    paths = _vault(tmp_path)
    rel = "notes/old.jsonl"
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"a": 1}\n', encoding="utf-8")

    old_processed = paths.archive_processed / "notes/old.md"
    old_index = paths.knowledge_index / "notes/old.md"
    old_processed.parent.mkdir(parents=True, exist_ok=True)
    old_index.parent.mkdir(parents=True, exist_ok=True)
    old_processed.write_text("# Old\n", encoding="utf-8")
    old_index.write_text(
        "---\ntitle: Old\ntype: source_note\nstatus: processed\nmy_key: keep-me\n---\n\n"
        "# Summary\n\nx\n",
        encoding="utf-8",
    )
    (paths.archive_processed / "notes/old_assets").mkdir(parents=True, exist_ok=True)
    (paths.archive_processed / "notes/old_assets/fig.png").write_bytes(b"f")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel, source_hash="stale-hash", size_bytes=1, extension=".jsonl",
        extractor="jsonl", status="processed", raw_path="archive/raw/" + rel,
        processed_path="archive/processed/notes/old.md",
        index_note_path="knowledge/index/notes/old.md",
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    ))

    plan = plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    run_ingest(paths, plan, dry_run=False, logger=_LOG)

    # One note per source, under today's name, with the user's key merged in.
    assert not old_processed.exists() and not old_index.exists()
    new_index = paths.knowledge_index / "notes/old.jsonl.md"
    assert new_index.is_file()
    assert "my_key: keep-me" in new_index.read_text(encoding="utf-8")
    assert (paths.archive_processed / "notes/old.jsonl.md").is_file()
    # The orphaned assets dir is gone too (migrated, then cleared by an
    # extraction that produces no assets).
    assert not (paths.archive_processed / "notes/old_assets").exists()


# ------------------------------------- created_at is a birthday (AUD-019)

def test_created_at_survives_a_reingest(tmp_path: Path) -> None:
    # AUD-019: created_at was reset to now on every re-ingest, so Now.md's
    # "Recently added sources" listed whatever was last re-extracted.
    paths = _vault(tmp_path)
    rel = "notes/a.txt"
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("new bytes\n", encoding="utf-8")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel, source_hash="stale-hash", size_bytes=1, extension=".txt",
        extractor="text", status="processed", raw_path="archive/raw/" + rel,
        processed_path="archive/processed/" + rel + ".md",
        index_note_path="knowledge/index/" + rel + ".md",
        created_at="2020-01-01T00:00:00Z", updated_at="2020-01-01T00:00:00Z",
    ))

    plan = plan_ingest(paths, sources=[paths.archive_raw], from_archive=True, logger=_LOG)
    run_ingest(paths, plan, dry_run=False, logger=_LOG)

    rec = latest_records_by_path(paths.metadata_index_jsonl)[rel]
    assert rec.created_at == "2020-01-01T00:00:00Z"
    assert rec.updated_at != "2020-01-01T00:00:00Z"


# --------------------------- a failed summary stays reachable (AUD-020)

def test_backfill_picks_up_a_failed_summary(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    # AUD-020: ingestion never revisits a processed record, so a record whose
    # summariser call failed (empty summary + the "summary: skipped" note) is
    # repaired only by --backfill-summaries. It must actually see it.
    from ingest_lib import pipeline as pl
    from ingest_lib.summarize import SummaryResult

    paths = _vault(tmp_path)
    rel = "notes/a.txt"
    processed = paths.archive_processed / (rel + ".md")
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text("# A\n\n---\n\nthe body\n", encoding="utf-8")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel, source_hash="h", size_bytes=1, extension=".txt",
        extractor="text", status="processed", raw_path="archive/raw/" + rel,
        processed_path="archive/processed/" + rel + ".md",
        index_note_path="knowledge/index/" + rel + ".md",
        notes=["summary: skipped (see warnings in log)"],
    ))

    monkeypatch.setattr(pl, "_summary_enabled", lambda: True)
    monkeypatch.setattr(
        pl, "_summarize",
        lambda *a, **k: SummaryResult(summary="a summary", key_points=["kp"],
                                     topics=["t"], notes=["summary: stub"]),
    )
    stats = pl.backfill_summaries(paths, logger=_LOG)

    assert stats.processed == 1
    rec = latest_records_by_path(paths.metadata_index_jsonl)[rel]
    assert rec.summary == "a summary" and rec.topics == ["t"]


# ------------------------ --path outside the vault keeps subfolders (AUD-021)

def test_path_outside_the_vault_keeps_its_subfolders(tmp_path: Path) -> None:
    # AUD-021: every file collapsed to its basename, so b/same.txt was
    # silently dropped as a duplicate of a/same.txt.
    paths = _vault(tmp_path)
    ext = tmp_path / "ext"
    (ext / "a").mkdir(parents=True)
    (ext / "b").mkdir(parents=True)
    (ext / "a/same.txt").write_text("A\n", encoding="utf-8")
    (ext / "b/same.txt").write_text("B\n", encoding="utf-8")

    plan = plan_ingest(paths, sources=[ext], from_archive=False, logger=_LOG)
    assert [i.relative_path for i in plan.items] == ["a/same.txt", "b/same.txt"]


def test_single_file_outside_the_vault_keeps_its_basename(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    lone = tmp_path / "elsewhere/report.txt"
    lone.parent.mkdir(parents=True)
    lone.write_text("body\n", encoding="utf-8")

    plan = plan_ingest(paths, sources=[lone], from_archive=False, logger=_LOG)
    assert [i.relative_path for i in plan.items] == ["report.txt"]


# ------------------------ a dropped index line is logged (AUD-022)

def test_malformed_index_line_is_logged_not_silent(tmp_path: Path,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    # AUD-022: a torn or hand-broken line made its source look un-ingested
    # (re-extract, re-summarise, duplicate record) with no trace anywhere.
    paths = _vault(tmp_path)
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path="a.txt", source_hash="h", size_bytes=1, extension=".txt",
        extractor="text", status="processed", raw_path="archive/raw/a.txt",
        processed_path=None, index_note_path=None,
    ))
    with paths.metadata_index_jsonl.open("a", encoding="utf-8") as fh:
        fh.write('{"relative_path": "b.txt", "source_hash"\n')
        fh.write('{"source_hash": "h2"}\n')  # parses, but not a record

    with caplog.at_level(logging.WARNING, logger="ingest_lib.metadata"):
        recs = list(iter_records(paths.metadata_index_jsonl))

    assert [r.relative_path for r in recs] == ["a.txt"]
    assert "line 2: unparseable JSON" in caplog.text
    assert "line 3: does not match the record schema" in caplog.text


# ------------------- same bytes under two paths are cross-linked (AUD-026)

def test_identical_content_under_two_paths_is_cross_referenced(tmp_path: Path) -> None:
    # AUD-026: the summary was deduplicated by hash but nothing said the two
    # notes describe the same bytes.
    paths = _vault(tmp_path)
    _drop(paths, "notes/a.txt", "identical body\n")
    _drop(paths, "notes/b.txt", "identical body\n")
    _ingest(paths)

    latest = latest_records_by_path(paths.metadata_index_jsonl)
    assert not any("duplicate of" in n for n in latest["notes/a.txt"].notes)
    assert any("duplicate of `notes/a.txt`" in n for n in latest["notes/b.txt"].notes)
    note = (paths.knowledge_index / "notes/b.txt.md").read_text(encoding="utf-8")
    assert "duplicate of `notes/a.txt`" in note


# ------------- trailing-space source stems get clean derived names (AUD-083)

def test_trailing_space_source_gets_clean_derived_names(tmp_path: Path) -> None:
    # AUD-083: two raw sources ended in a space before the extension. Every
    # wikilink parser strips whitespace inside [[...]], so nothing could ever
    # link to the derived note, index note or assets dir that inherited it
    # (and the names are illegal on Windows). archive/raw keeps the name; the
    # derived side is sanitised.
    paths = _vault(tmp_path)
    _drop(paths, "notes/report .txt", "body\n")
    assert _ingest(paths).processed == 1

    rec = latest_records_by_path(paths.metadata_index_jsonl)["notes/report .txt"]
    # The index key — the source's own path — is untouched.
    assert rec.raw_path == "archive/raw/notes/report .txt"
    assert (paths.root / rec.raw_path).is_file()
    assert rec.processed_path == "archive/processed/notes/report.txt.md"
    assert rec.index_note_path == "knowledge/index/notes/report.txt.md"
    assert (paths.root / rec.processed_path).is_file()
    assert (paths.root / rec.index_note_path).is_file()

    note = (paths.root / rec.index_note_path).read_text(encoding="utf-8")
    # Both generated links survive a wikilink parser's .strip(): the derived
    # note by its sanitised name, the immutable source by its full name.
    assert "[[archive/processed/notes/report.txt.md]]" in note
    assert "[[archive/raw/notes/report .txt]]" in note
    for link in _wikilinks(note):
        assert link == link.strip()
        assert (paths.root / link).exists() or (paths.root / f"{link}.md").is_file()


def test_odd_interior_whitespace_survives_sanitising(tmp_path: Path) -> None:
    # Only the surrounding whitespace is a defect: an interior double space
    # resolves fine and the existing concept-note links reproduce it, so
    # collapsing runs would rename files no defect requires renaming.
    paths = _vault(tmp_path)
    _drop(paths, "notes/ENT overview  Jan26 RW .txt", "body\n")
    assert _ingest(paths).processed == 1

    rec = latest_records_by_path(paths.metadata_index_jsonl)[
        "notes/ENT overview  Jan26 RW .txt"
    ]
    assert rec.processed_path == "archive/processed/notes/ENT overview  Jan26 RW.txt.md"
    assert rec.index_note_path == "knowledge/index/notes/ENT overview  Jan26 RW.txt.md"
    assert (paths.root / rec.processed_path).is_file()
    assert (paths.root / rec.index_note_path).is_file()
    assert derived_assets_dirname("notes/ENT overview  Jan26 RW .txt") == (
        "ENT overview  Jan26 RW.txt_assets"
    )


@pytest.mark.parametrize(("name", "expected"), [
    ("report .pdf", "report.pdf"),      # the AUD-083 shape
    ("report.pdf ", "report.pdf"),      # trailing space after the extension
    (" report.pdf", "report.pdf"),      # leading space, stripped by parsers too
    ("README ", "README"),              # extensionless
    (".env ", ".env"),                  # dotfile: no stem to strip
    ("a  b.pdf", "a  b.pdf"),           # interior whitespace is kept
    ("report.pdf", "report.pdf"),       # already clean
])
def test_sanitise_derived_name(name: str, expected: str) -> None:
    assert sanitise_derived_name(name) == expected
