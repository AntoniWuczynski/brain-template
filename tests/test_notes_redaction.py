"""Every note ingestion writes goes through redaction.

Until 2026-09-18 only the two transcript connectors redacted, so a credential
inside a PDF, a meeting export, a dropped file or a vision extraction reached
`archive/processed/` in clear text. The writers redact now, so a path does
not have to opt in to be covered.

The note profile applies here: a git SHA or a URL inside an extracted document
is a reference worth keeping, not a credential worth destroying.
"""
from __future__ import annotations

from pathlib import Path

from ingest_lib.notes import NoteContent, write_index_note, write_processed_note

_SHA = "e83c5163316f89bfbde7d9ab23ca2e25604af290"


def _content(**over: object) -> NoteContent:
    base: dict[str, object] = {
        "title": "A document",
        "source_relative_path": "archive/raw/docs/a.pdf",
        "source_hash": "a" * 64,
        "status": "processed",
        "extracted_markdown": "body",
        "processing_notes": ["extracted with pypdf"],
        "extractor": "pypdf",
    }
    base.update(over)
    return NoteContent(**base)  # type: ignore[arg-type]  # kwargs built dynamically for the fixture


def test_processed_note_redacts_a_credential_in_the_body(tmp_path: Path) -> None:
    target = tmp_path / "a.md"
    write_processed_note(
        target=target,
        content=_content(extracted_markdown="the key is AKIA1234567890ABCDEF ok"),
    )
    written = target.read_text(encoding="utf-8")
    assert "AKIA1234567890ABCDEF" not in written
    assert "[REDACTED:aws-access-key]" in written


def test_processed_note_redacts_the_summary_and_processing_notes(tmp_path: Path) -> None:
    target = tmp_path / "b.md"
    write_processed_note(
        target=target,
        content=_content(
            summary="contains sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa here",
            processing_notes=["failed: Bearer abcdefghijklmnopqrstuvwxyz123456"],
        ),
    )
    written = target.read_text(encoding="utf-8")
    assert "sk-ant-api03" not in written
    assert "abcdefghijklmnopqrstuvwxyz123456" not in written


def test_processed_note_keeps_the_source_hash_intact(tmp_path: Path) -> None:
    """The hash is 64 hex and must survive: it is how a note is joined back to
    its index record, and rewriting it would be silent corruption."""
    target = tmp_path / "c.md"
    write_processed_note(target=target, content=_content(source_hash="b" * 64))
    assert "b" * 64 in target.read_text(encoding="utf-8")


def test_processed_note_keeps_a_git_sha_in_the_body(tmp_path: Path) -> None:
    target = tmp_path / "d.md"
    write_processed_note(
        target=target, content=_content(extracted_markdown=f"fixed in {_SHA} upstream")
    )
    assert _SHA in target.read_text(encoding="utf-8")


def test_index_note_redacts_too(tmp_path: Path) -> None:
    target = tmp_path / "e.md"
    write_index_note(
        target=target,
        content=_content(summary="token AKIA1234567890ABCDEF in the summary"),
    )
    assert "AKIA1234567890ABCDEF" not in target.read_text(encoding="utf-8")
