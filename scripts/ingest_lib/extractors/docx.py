"""DOCX extractor (paragraphs + tables, in document order)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .base import ExtractionResult


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    try:
        from docx import Document  # type: ignore[import-not-found]
        from docx.oxml.ns import qn  # type: ignore[import-not-found]
    except ImportError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="docx",
            markdown="",
            error=f"python-docx missing: {exc}",
        )

    try:
        doc = Document(str(src))
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(
            status="manual_review",
            extractor="docx",
            markdown="",
            error=f"open failed: {exc}",
        )

    parts: list[str] = []
    body = doc.element.body
    _walk_block_children(body, qn, parts)
    # _walk_block_children only ever appends strings (never None), so no
    # None-filter is needed here.
    markdown = "\n\n".join(parts) + "\n"

    # Embedded images/drawings and OLE objects carry no extractable text, so
    # a doc that has them is only PARTIALLY captured. Say so honestly and
    # downgrade rather than claiming full extraction (the idempotency skip
    # would otherwise make the silent loss permanent).
    n_drawings = sum(1 for _ in body.iter(qn("w:drawing")))
    n_pict = sum(1 for _ in body.iter(qn("w:pict")))
    n_obj = sum(1 for _ in body.iter(qn("w:object")))
    n_visual = n_drawings + n_pict + n_obj
    notes: list[str] = []
    status: Literal["processed", "partial", "manual_review"] = "processed"
    if n_visual:
        notes.append(
            f"{n_visual} embedded image(s)/drawing(s)/object(s) not extracted "
            "(text captured; visual content is not)"
        )
        status = "partial"

    return ExtractionResult(
        status=status,
        extractor="docx",
        markdown=markdown,
        notes=notes,
    )


def _walk_block_children(container, qn, parts: list[str]) -> None:
    """Append Markdown for a container's block children in document order.

    Descends into ``w:sdt`` content controls (TOCs, structured-document
    tags), whose body would otherwise be dropped — they are siblings of
    ``w:p``/``w:tbl``, not inside them. Paragraph text uses a recursive
    ``w:t`` scan, so text-box (``w:txbxContent``) prose is captured too."""
    for child in container.iterchildren():
        tag = child.tag
        if tag == qn("w:p"):
            text = "".join(t.text or "" for t in child.iter(qn("w:t")))
            parts.append(text if text.strip() else "")
        elif tag == qn("w:tbl"):
            parts.append(_table_to_markdown(child))
        elif tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                _walk_block_children(content, qn, parts)


def _cell_text(cell, qn) -> str:
    """Cell text, escaped for a Markdown table cell: a literal '|' would add
    a phantom column, a newline would break the row."""
    text = "".join(t.text or "" for t in cell.iter(qn("w:t"))).strip()
    return text.replace("|", "\\|").replace("\n", " ")


def _table_to_markdown(tbl_elem) -> str:
    from docx.oxml.ns import qn  # type: ignore[import-not-found]

    rows: list[list[str]] = []
    # DIRECT children only (findall, not iter): a nested table's rows/cells
    # would otherwise be pulled into the outer table twice and inflate the
    # column count.
    for row in tbl_elem.findall(qn("w:tr")):
        cells = [_cell_text(cell, qn) for cell in row.findall(qn("w:tc"))]
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    sep = ["---"] * width
    body = rows[1:] if len(rows) > 1 else []
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)
