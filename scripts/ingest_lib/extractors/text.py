"""Plain-text and source-code extractor."""
from __future__ import annotations

from pathlib import Path

from .base import ExtractionResult, fence


_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB safety cap; oversize files marked partial


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    try:
        size = src.stat().st_size
    except OSError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="text",
            markdown="",
            error=f"stat failed: {exc}",
        )

    truncated = size > _MAX_BYTES
    try:
        # Read BYTES then decode, so the cap is a true byte cap: a text-mode
        # read(_MAX_BYTES) reads that many CHARACTERS, disagreeing with the
        # byte-based ``truncated`` flag on any multibyte file.
        with src.open("rb") as fh:
            data = fh.read(_MAX_BYTES)
        text = data.decode("utf-8", errors="replace")
    except OSError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="text",
            markdown="",
            error=f"read failed: {exc}",
        )

    suffix = src.suffix.lstrip(".").lower() or "text"
    # Dynamic fence: source files (esp. .md/.rst) routinely contain ``` runs.
    fenced = fence(text, suffix) + "\n"

    notes = []
    if truncated:
        notes.append(f"file truncated to {_MAX_BYTES} bytes (size was {size})")

    return ExtractionResult(
        status="partial" if truncated else "processed",
        extractor="text",
        markdown=fenced,
        notes=notes,
    )
