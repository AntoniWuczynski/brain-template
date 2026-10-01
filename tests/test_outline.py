"""ingest_lib.outline: section previews skip heading lines and stay short."""
from __future__ import annotations

from ingest_lib.outline import PREVIEW_CHARS, build_outline
from ingest_lib.semantic_meta import MetaRow


def _row(idx: int, heading: str, text: str) -> MetaRow:
    return MetaRow(
        source_relative_path="s.pdf", source_hash="h", title="T", chunk_idx=idx,
        text=text, origin="archive", heading_path=heading, model="", gen="",
    )


def test_preview_skips_headings_and_collapses_whitespace() -> None:
    out = build_outline([
        _row(0, "Demo", "## Demo\n\nPrim's   algorithm\ngrows one tree"),
        _row(1, "Demo", "more"),
        _row(2, "Key idea", "# Key idea\n" + "y" * 500),
    ])
    assert [(s.heading_path, s.first_chunk, s.last_chunk) for s in out] == [
        ("Demo", 0, 1), ("Key idea", 2, 2),
    ]
    assert out[0].preview == "Prim's algorithm grows one tree"
    assert out[1].preview == "y" * PREVIEW_CHARS
