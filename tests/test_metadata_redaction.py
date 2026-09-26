"""`metadata/index.jsonl` is redacted too.

The index record is built from the summariser's output directly, not from the
note, so redacting the note does not cover it. The file is git-tracked, and
`metadata` is in the MCP read allowlist, so a credential quoted into a summary
would be committed, pushed and readable over a tool call.
"""
from __future__ import annotations

from pathlib import Path

from ingest_lib.metadata import IndexRecord, append_record, iter_records


def _record(**over: object) -> IndexRecord:
    base: dict[str, object] = {
        "relative_path": "docs/a.pdf",
        "source_hash": "c" * 64,
        "size_bytes": 10,
        "extension": ".pdf",
        "extractor": "pypdf",
        "status": "processed",
        "raw_path": "archive/raw/docs/a.pdf",
        "processed_path": "archive/processed/docs/a.pdf.md",
        "index_note_path": None,
    }
    base.update(over)
    return IndexRecord(**base)  # type: ignore[arg-type]  # kwargs built dynamically for the fixture


def test_index_record_summary_is_redacted(tmp_path: Path) -> None:
    jsonl = tmp_path / "index.jsonl"
    append_record(jsonl, _record(summary="the key is AKIA1234567890ABCDEF"))
    written = jsonl.read_text(encoding="utf-8")
    assert "AKIA1234567890ABCDEF" not in written
    assert "[REDACTED:aws-access-key]" in written


def test_index_record_key_points_are_redacted(tmp_path: Path) -> None:
    jsonl = tmp_path / "index.jsonl"
    append_record(jsonl, _record(key_points=["uses sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa"]))
    assert "sk-ant-api03" not in jsonl.read_text(encoding="utf-8")


def test_index_record_source_hash_survives(tmp_path: Path) -> None:
    """The hash is the join key for idempotency and the integrity sweep."""
    jsonl = tmp_path / "index.jsonl"
    append_record(jsonl, _record(source_hash="d" * 64))
    rec = next(iter_records(jsonl))
    assert rec.source_hash == "d" * 64
