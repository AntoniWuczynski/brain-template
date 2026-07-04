"""Jupyter notebook extractor: code + markdown cells, no rich outputs."""
from __future__ import annotations

from pathlib import Path

from .base import ExtractionResult, fence


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    try:
        import nbformat  # type: ignore[import-not-found]
    except ImportError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="notebook",
            markdown="",
            error=f"nbformat missing: {exc}",
        )

    try:
        # as_version=4 upgrades legacy v3 notebooks in memory (v3 exposes
        # `worksheets`, not `cells`, so NO_CONVERT would AttributeError on a
        # perfectly convertible file). Genuinely corrupt files still raise.
        nb = nbformat.read(str(src), as_version=4)
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(
            status="manual_review",
            extractor="notebook",
            markdown="",
            error=f"open failed: {exc}",
        )

    parts: list[str] = []
    skipped_outputs = 0
    for i, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "markdown":
            parts.append(cell.source.strip())
        elif cell.cell_type == "code":
            lang = nb.metadata.get("kernelspec", {}).get("language", "python")
            parts.append(fence(cell.source, lang))
            outs = cell.get("outputs", [])
            if outs:
                skipped_outputs += len(outs)
        elif cell.cell_type == "raw":
            parts.append(fence(cell.source))
        else:
            parts.append(f"_(unknown cell type: `{cell.cell_type}` at index {i})_")
        parts.append("")

    notes: list[str] = []
    if skipped_outputs:
        notes.append(
            f"omitted {skipped_outputs} cell output(s) "
            "(images/text/streams) — see source notebook for them"
        )
    return ExtractionResult(
        status="processed",
        extractor="notebook",
        markdown="\n".join(parts),
        notes=notes,
    )
