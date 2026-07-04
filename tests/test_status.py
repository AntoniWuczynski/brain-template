"""Processing Dashboard + Manual Review generation (P1)."""
from __future__ import annotations

import logging
from pathlib import Path

from ingest_lib.config import paths_for_root
from ingest_lib.metadata import IndexRecord, append_record
from ingest_lib.status import rebuild_status

_LOG = logging.getLogger("test")


def _rec(rel: str, status: str, *, extractor: str, ext: str, src_hash: str,
         error: str | None = None, raw: str | None = None) -> IndexRecord:
    return IndexRecord(
        relative_path=rel, source_hash=src_hash, size_bytes=1, extension=ext,
        extractor=extractor, status=status,  # type: ignore[arg-type]
        raw_path=raw or f"archive/raw/{rel}", processed_path=None,
        index_note_path=None, error=error,
    )


def _vault(tmp_path: Path):
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def test_dashboard_and_review_content(tmp_path: Path):
    paths = _vault(tmp_path)
    append_record(paths.metadata_index_jsonl,
                  _rec("uni/a.pdf", "processed", extractor="pdf-mineru", ext=".pdf", src_hash="h1"))
    append_record(paths.metadata_index_jsonl,
                  _rec("uni/b.pdf", "partial", extractor="pdf-pypdf", ext=".pdf", src_hash="h2",
                       error="MinerU absent"))
    append_record(paths.metadata_index_jsonl,
                  _rec("uni/c.docx", "manual_review", extractor="docx", ext=".docx", src_hash="h3",
                       error="open failed", raw="archive/failed/uni/c.docx"))

    st = rebuild_status(paths, logger=_LOG)
    assert st.dashboard_written and st.review_written
    assert st.needs_review == 2

    dash = (paths.knowledge_index / "Processing Dashboard.md").read_text(encoding="utf-8")
    assert "| processed | 1 |" in dash
    assert "| partial | 1 |" in dash
    assert "| manual_review | 1 |" in dash
    assert "pdf-mineru" in dash and "docx" in dash

    review = (paths.knowledge_index / "Manual Review.md").read_text(encoding="utf-8")
    assert "uni/b.pdf" in review and "MinerU absent" in review
    assert "uni/c.docx" in review and "open failed" in review
    # Partial rows carry the exact retry command.
    assert "--path archive/raw/uni/b.pdf" in review


def test_inbox_pending_vs_ingested(tmp_path: Path):
    paths = _vault(tmp_path)
    # A file whose hash matches an ingested record -> "already ingested".
    ingested = paths.inbox / "done.txt"
    ingested.write_text("already here\n", encoding="utf-8")
    from ingest_lib.hashing import sha256_of
    append_record(paths.metadata_index_jsonl,
                  _rec("done.txt", "processed", extractor="text", ext=".txt",
                       src_hash=sha256_of(ingested)))
    # A file with no matching record -> pending.
    (paths.inbox / "new.pdf").write_text("brand new\n", encoding="utf-8")

    st = rebuild_status(paths, logger=_LOG)
    assert st.inbox_pending == 1
    assert st.inbox_ingested == 1
    dash = (paths.knowledge_index / "Processing Dashboard.md").read_text(encoding="utf-8")
    assert "new.pdf" in dash                       # pending listed in detail
    assert "1 inbox file(s) match an ingested hash" in dash


def test_rebuild_is_skip_unchanged(tmp_path: Path):
    paths = _vault(tmp_path)
    append_record(paths.metadata_index_jsonl, _rec("a.txt", "processed", extractor="text", ext=".txt", src_hash="h"))
    first = rebuild_status(paths, logger=_LOG)
    assert first.dashboard_written
    second = rebuild_status(paths, logger=_LOG)
    assert not second.dashboard_written and not second.review_written


def test_user_tail_preserved(tmp_path: Path):
    paths = _vault(tmp_path)
    append_record(paths.metadata_index_jsonl, _rec("a.txt", "processed", extractor="text", ext=".txt", src_hash="h"))
    rebuild_status(paths, logger=_LOG)
    target = paths.knowledge_index / "Processing Dashboard.md"
    text = target.read_text(encoding="utf-8")
    text = text.replace(
        "_(Your hand-written notes go here. Preserved across re-runs.)_",
        "MY IMPORTANT NOTE",
    )
    target.write_text(text, encoding="utf-8")
    # A content change (new record) forces a rewrite; the tail must survive.
    append_record(paths.metadata_index_jsonl, _rec("b.txt", "processed", extractor="text", ext=".txt", src_hash="h2"))
    rebuild_status(paths, logger=_LOG)
    assert "MY IMPORTANT NOTE" in target.read_text(encoding="utf-8")
