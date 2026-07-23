"""Standalone-image extractor: photos, whiteboards, screenshots, scans.

A bare ``.jpg``/``.png``/``.heic`` dropped in the inbox was silently skipped
(no image extension was registered). This runs each image through a
vision-LLM — reusing ``vlm.py``'s provider dispatch, output-cap handling and
honest failure/blank semantics — to produce a searchable transcription /
description. Honest by construction: no vision backend configured means
``manual_review`` (never a hallucinated caption), exactly like ``vlm.py``.

The original image already lives immutably in ``archive/raw/`` and the index
note links back to it, so nothing is duplicated as an asset — the extracted
text is the value added.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Final

from .base import ExtractionResult
from .. import summarize as _summ
from . import vlm as _vlm

_LOG = logging.getLogger(__name__)

# Raster formats Pillow decodes out of the box. HEIC/HEIF (iPhone default)
# needs the optional ``pillow-heif`` plugin; handled gracefully below.
IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif",
    ".heic", ".heif",
)

_PROMPT: Final[str] = (
    "You are transcribing ONE image into Markdown for a searchable knowledge "
    "vault. The image may be a photo, a whiteboard, a screenshot, a scanned "
    "page or handwritten notes. Rules:\n"
    "- Transcribe ALL legible text verbatim and faithfully, preserving "
    "headings, lists and emphasis. Use LaTeX for real mathematics only "
    "($...$ inline, $$...$$ display).\n"
    "- For a diagram, chart, photo or sketch with little text, write a "
    "concise `[image: <what it depicts>]` describing what is actually shown "
    "— do NOT attempt ASCII art.\n"
    "- LAYOUT: if the page is laid out in columns, transcribe each column as "
    "its own block under its own sub-heading. Do NOT interleave columns line "
    "by line — that invents pairings the page does not make. Where a diagram "
    "aligns labels against an axis or timeline, state which label sits "
    "against which axis value.\n"
    "- ROTATED TEXT: text along a margin or axis may be rotated 90 degrees. "
    "Read it in its own orientation, and work out which of the two possible "
    "rotations is correct before reading — the wrong one turns '2026' into "
    "'9202' and swaps digit order. Say nothing about a rotated number you "
    "have not resolved this way; mark it `[illegible]` instead.\n"
    "- DELETIONS: text the author struck through is deleted. Render it as "
    "`~~struck~~`, never as live content. If you cannot read what was struck "
    "out, write `~~[illegible]~~` — do not guess the deleted word.\n"
    "- SHOW-THROUGH: photographed paper often shows faint mirrored writing "
    "bleeding through from the reverse of the sheet. It is not on this page. "
    "Ignore it entirely.\n"
    "- If something is unreadable, write `[illegible]`. NEVER guess or invent "
    "text, numbers, names or content that is not actually in the image. "
    "Inventing a plausible date or name is the worst failure mode there is: "
    "when a token is doubtful, transcribe what you can and mark the rest "
    "`[illegible]`. Honesty over completeness.\n"
    "Output ONLY the Markdown transcription/description, no preamble."
)


def _load_png_bytes(src: Path) -> bytes:
    """Decode ``src`` and re-encode as PNG bytes for the vision API. Raises on
    an undecodable/corrupt image; HEIC without the plugin raises too."""
    from PIL import Image  # type: ignore[import-not-found]

    if src.suffix.lower() in (".heic", ".heif"):
        try:
            import pillow_heif  # type: ignore[import-not-found]
            pillow_heif.register_heif_opener()
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "HEIC/HEIF needs the optional 'pillow-heif' package "
                "(uv pip install pillow-heif)"
            ) from exc
    with Image.open(src) as im:
        im.load()
        rgb = im.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    """Transcribe/describe a single image with the vision LLM."""
    provider = _summ._select_provider()
    if provider is None:
        return ExtractionResult(
            status="manual_review",
            extractor="image-vlm",
            markdown="",
            error="image: no vision LLM provider configured (set ANTHROPIC_API_KEY "
            "or BRAIN_LLM_PROVIDER)",
        )

    import os
    model = os.environ.get("BRAIN_VLM_MODEL") or _vlm._default_vlm_model(provider)

    try:
        png = _load_png_bytes(src)
    except Exception as exc:  # noqa: BLE001 — decode failure is per-file
        return ExtractionResult(
            status="manual_review",
            extractor="image-vlm",
            markdown="",
            error=f"image: could not decode ({exc!r})",
        )

    result = _vlm._transcribe_page(
        png=png, provider=provider, model=model, page_no=1, prompt=_PROMPT
    )
    notes = [f"image-vlm: {provider}/{model}"]
    if not isinstance(result, _vlm._VisionText):
        reason = result.reason if isinstance(result, _vlm._PageFailure) else "unknown"
        return ExtractionResult(
            status="manual_review",
            extractor="image-vlm",
            markdown="",
            error=f"image: vision transcription failed ({reason})",
            notes=notes,
        )

    body = result.text.strip() or "_(no legible content in image)_"
    if result.truncated:
        body += (
            "\n\n_(transcription truncated at the model output cap — see the "
            "original image)_"
        )
    return ExtractionResult(
        status="partial" if result.truncated else "processed",
        extractor="image-vlm",
        markdown=body + "\n",
        notes=notes,
    )
