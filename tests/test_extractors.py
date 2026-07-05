"""Extractor unit tests over small synthetic fixtures.

The extractors package had no coverage despite holding the riskiest
branching in the pipeline (fence collisions, table escaping, honesty
downgrades). These pin the correctness fixes without needing MinerU,
torch, or a vision API.
"""
from __future__ import annotations

import base64
from pathlib import Path


from ingest_lib.extractors.base import fence
from ingest_lib.extractors import dataset as ds
from ingest_lib.extractors import text as text_ex


# --------------------------------------------------------------- fence()

def test_fence_plain_uses_three_backticks():
    out = fence("hello", "py")
    assert out == "```py\nhello\n```"


def test_fence_grows_past_inner_backtick_run():
    # Content containing ``` must be wrapped in a LONGER fence, else it
    # closes early.
    content = "before\n```\ninner\n```\nafter"
    out = fence(content, "md")
    assert out.startswith("````md\n")
    assert out.endswith("\n````")
    # The inner ``` no longer terminates the block.
    assert out.count("````") == 2


# ---------------------------------------------------------------- text.py

def test_text_extractor_survives_embedded_code_fence(tmp_path: Path):
    src = tmp_path / "note.md"
    src.write_text("# Title\n\n```python\nprint('x')\n```\n", encoding="utf-8")
    res = text_ex.extract(src, tmp_path / "assets")
    assert res.status == "processed"
    # The wrapping fence is longer than the inner ``` so nothing spills out.
    assert res.markdown.startswith("````md\n")
    assert "print('x')" in res.markdown


def test_text_extractor_byte_cap_is_true_bytes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(text_ex, "_MAX_BYTES", 8)
    src = tmp_path / "big.txt"
    src.write_text("abcdefghijklmnop", encoding="utf-8")  # 16 bytes
    res = text_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert any("truncated" in n for n in res.notes)


# ------------------------------------------------------------- dataset.py

def test_dataset_csv_escapes_pipe_in_header(tmp_path: Path):
    src = tmp_path / "d.csv"
    src.write_text("price|usd,name\n10,widget\n", encoding="utf-8")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "processed"
    # The '|' in the header is escaped so it can't add a phantom column.
    assert "price\\|usd" in res.markdown


def test_dataset_jsonl_marks_partial_on_unparseable_lines(tmp_path: Path):
    src = tmp_path / "d.jsonl"
    src.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert any("unparseable" in n for n in res.notes)
    assert "**Records:** 2" in res.markdown


def test_dataset_csv_oversized_field_is_manual_review(tmp_path: Path):
    import csv
    src = tmp_path / "d.csv"
    big = "x" * (csv.field_size_limit() + 10)
    src.write_text(f'col\n"{big}"\n', encoding="utf-8")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "csv parse failed" in (res.error or "")


# ------------------------------------------------------- docx (D8/F110)

# 1x1 transparent PNG for embedding.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_docx_extracts_table_and_flags_embedded_image(tmp_path: Path):
    from docx import Document
    from docx.oxml import OxmlElement
    doc = Document()
    doc.add_paragraph("Intro paragraph.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "price|usd"   # pipe must be escaped
    table.cell(0, 1).text = "name"
    table.cell(1, 0).text = "10"
    table.cell(1, 1).text = "widget"
    # Inject a w:drawing element (what the extractor counts) rather than a
    # real image, so the test doesn't depend on python-docx's image parser.
    run = doc.add_paragraph("figure below").add_run()
    run._r.append(OxmlElement("w:drawing"))
    src = tmp_path / "d.docx"
    doc.save(str(src))

    from ingest_lib.extractors import docx as docx_ex
    res = docx_ex.extract(src, tmp_path / "a")
    assert "Intro paragraph." in res.markdown
    assert "price\\|usd" in res.markdown            # table extracted + escaped
    assert "| widget |" in res.markdown
    # An embedded image can't be extracted -> honest partial + note.
    assert res.status == "partial"
    assert any("image" in n or "drawing" in n for n in res.notes)


# ------------------------------------------------------- pptx (D8/F036)

def test_pptx_extracts_table_and_flags_picture(tmp_path: Path):
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tb = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    tb.text_frame.text = "A bullet of text"
    tbl = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    tbl.cell(0, 0).text = "col|a"
    tbl.cell(0, 1).text = "colb"
    tbl.cell(1, 0).text = "1"
    tbl.cell(1, 1).text = "2"
    import io
    slide.shapes.add_picture(io.BytesIO(_PNG), Inches(1), Inches(3))
    src = tmp_path / "p.pptx"
    prs.save(str(src))

    from ingest_lib.extractors import pptx as pptx_ex
    res = pptx_ex.extract(src, tmp_path / "a")
    assert "A bullet of text" in res.markdown
    assert "col\\|a" in res.markdown                # table cell extracted + escaped
    assert "| colb |" in res.markdown
    assert res.status == "partial"                  # picture present
    assert any("picture" in n for n in res.notes)
