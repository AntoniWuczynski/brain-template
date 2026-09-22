"""Notes written over MCP are redacted before they reach disk.

`mcp_server` never imports `ingest_lib.pipeline`, so the ingestion-side
redaction added alongside this does not cover tool calls at all: every write
tool lands on `_atomic_write_text`. That matters more than the ingestion path
here, because a tool call is how the new clients write — and whatever they
write is committed and pushed automatically.

The note profile applies: an agent writing a note about a commit should get
its SHA back intact.
"""
from __future__ import annotations

from pathlib import Path

from mcp_server.tools import _atomic_write_text

_SHA = "e83c5163316f89bfbde7d9ab23ca2e25604af290"
_HASH = "a" * 64


def test_mcp_write_redacts_a_credential(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    _atomic_write_text(target, "# Note\n\nkey AKIA1234567890ABCDEF here\n")
    written = target.read_text(encoding="utf-8")
    assert "AKIA1234567890ABCDEF" not in written
    assert "[REDACTED:aws-access-key]" in written


def test_mcp_write_redacts_a_bearer_token(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    _atomic_write_text(target, "curl -H 'Authorization: Bearer " + "f" * 40 + "'\n")
    assert "f" * 40 not in target.read_text(encoding="utf-8")


def test_mcp_write_keeps_frontmatter_hashes(tmp_path: Path) -> None:
    """Frontmatter carries source_hash, 64 hex. Rewriting it would silently
    break the join between a note and its index record."""
    target = tmp_path / "note.md"
    _atomic_write_text(target, f"---\nsource_hash: {_HASH}\n---\n\nbody\n")
    assert _HASH in target.read_text(encoding="utf-8")


def test_mcp_write_keeps_a_git_sha(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    _atomic_write_text(target, f"Fixed in {_SHA}.\n")
    assert _SHA in target.read_text(encoding="utf-8")
