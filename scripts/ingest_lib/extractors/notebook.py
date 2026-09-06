"""Jupyter notebook extractor: code + markdown cells, no rich outputs."""
from __future__ import annotations

from pathlib import Path

from .base import ExtractionResult, fence


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    try:
        import nbformat
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
        #
        # The ignore is the one unavoidable one in the repo (AUD-077).
        # nbformat 5.10.4 ships a py.typed marker, so mypy reads its source
        # rather than a stub — and that source is unannotated: `read`,
        # `reads`, `converter.convert` and `notebooknode.from_dict` all
        # declare bare `(fp, as_version, ...)`. So every entry point into the
        # library is an untyped call, there is no `types-nbformat` in
        # typeshed to supply signatures, and the error is on the CALL, not
        # the return — a Protocol over the returned NotebookNode could not
        # drop it. Laundering `nbformat.read` through a typed alias would
        # only hide the untypedness behind an unverifiable claim, so the
        # honest `no-untyped-call` ignore stays. Its cost is that `nb` is
        # Any: the cell loop below is unchecked, and the `except` around
        # this read is what turns a shape surprise into manual_review.
        nb = nbformat.read(str(src), as_version=4)  # type: ignore[no-untyped-call]
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
