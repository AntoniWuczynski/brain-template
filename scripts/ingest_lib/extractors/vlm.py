"""Vision-LLM PDF extractor for handwritten / scanned notes.

MinerU's pipeline OCR (PaddleOCR + UniMerNet) is built for *printed*
documents. On handwriting it does two bad things: it OCRs prose only
~85% accurately, and its formula model hallucinates dense fake LaTeX on
handwriting regions it misreads as math — fabricating equations that
were never on the page. That violates the vault's honest-extraction
rule, so handwritten material goes through a vision-LLM instead.

Each page is rendered to an image and sent to a vision model with a
strict transcription prompt (verbatim, LaTeX for real math, ``[diagram:
...]`` for figures, ``[illegible]`` for unreadable bits, never invent).
The rendered page image is also kept as an asset so the original
handwriting/diagrams remain viewable from the note.

Routing: the PDF extractor uses this module when ``BRAIN_PDF_EXTRACTOR``
is ``vlm`` (set it for handwritten modules). Provider/model selection
reuses the summarizer's config; the vision model defaults to
``claude-sonnet-4-6`` for anthropic and is overridable with
``BRAIN_VLM_MODEL``.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import shutil
from pathlib import Path
from typing import Final, Literal

from .base import ExtractionResult
from .. import summarize as _summ


_LOG = logging.getLogger(__name__)

# Render scale (1.0 ≈ 72 dpi). 2.0 was enough for clean handwriting in
# testing; bump via BRAIN_VLM_SCALE for finer script.
_DEFAULT_SCALE: Final[float] = 2.0
_MAX_OUTPUT_TOKENS: Final[int] = 4096

# Vision-capable defaults per provider. Override with BRAIN_VLM_MODEL.
_DEFAULT_VLM_MODELS: Final[dict[str, str]] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "local": os.environ.get("BRAIN_LOCAL_MODEL") or "llama3.2-vision",
}

_PROMPT: Final[str] = (
    "You are transcribing ONE page of handwritten lecture notes into "
    "Markdown for a searchable knowledge vault. Rules:\n"
    "- Transcribe all legible text verbatim and faithfully; preserve "
    "headings, lists and emphasis.\n"
    "- Use LaTeX for real mathematics only: $...$ inline, $$...$$ display. "
    "Transcribe Dirac/bra-ket notation faithfully (e.g. $\\langle\\psi| "
    "H |\\psi\\rangle$).\n"
    "- For any diagram, figure, circuit or sketch, write a concise "
    "`[diagram: <what it depicts>]` in place — do NOT attempt ASCII art.\n"
    "- If something is unreadable, write `[illegible]`. NEVER guess or "
    "invent text, equations, names or numbers that are not legibly on the "
    "page. Honesty over completeness.\n"
    "Output ONLY the Markdown transcription, with no preamble or commentary."
)


def extract(src: Path, assets_dir: Path) -> ExtractionResult:
    """Render every page and transcribe it with a vision model."""
    provider = _summ._select_provider()
    if provider is None:
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-vlm",
            markdown="",
            error="vlm: no LLM provider configured (set ANTHROPIC_API_KEY or "
            "BRAIN_LLM_PROVIDER)",
        )
    model = os.environ.get("BRAIN_VLM_MODEL") or _DEFAULT_VLM_MODELS.get(
        provider, _DEFAULT_VLM_MODELS["anthropic"]
    )

    try:
        pages_png = _render_pages(src)
    except Exception as exc:  # noqa: BLE001 — render failure is per-document
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-vlm",
            markdown="",
            error=f"vlm: page render failed ({exc!r})",
        )
    if not pages_png:
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-vlm",
            markdown="",
            error="vlm: PDF had no renderable pages",
        )

    assets_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    saved_assets: list[Path] = []
    failed_pages: list[int] = []

    for i, png in enumerate(pages_png, start=1):
        # Persist the rendered page so the original handwriting/diagrams stay
        # viewable from the note, and link it under the transcription.
        asset = assets_dir / f"page-{i:03d}.png"
        asset.write_bytes(png)
        saved_assets.append(asset)
        rel = f"{assets_dir.name}/{asset.name}"

        text = _transcribe_page(
            png=png, provider=provider, model=model, page_no=i
        )
        if text is None:
            # None == the API call FAILED (exception / structural error).
            failed_pages.append(i)
            body = "_(transcription failed for this page — see rendered image)_"
        else:
            # An empty string is a genuinely blank page (the vision helpers
            # return "" on success, None only on failure), so it is NOT a
            # failed page and must not downgrade the whole document.
            body = text.strip() or "_(blank page)_"
        sections.append(f"## Page {i}\n\n{body}\n\n![Page {i}]({rel})")

    markdown = "\n\n".join(sections) + "\n"
    notes = [
        f"vlm: {provider}/{model}, {len(pages_png)} page(s) transcribed",
        "handwritten/scanned source — figures kept as rendered page images",
    ]
    status: Literal["processed", "partial", "manual_review"]
    if failed_pages and len(failed_pages) == len(pages_png):
        # Every page failed → the pipeline discards this result and moves the
        # raw file to archive/failed, so the rendered page PNGs would be
        # orphaned under archive/processed. Clean them up and return no assets.
        shutil.rmtree(assets_dir, ignore_errors=True)
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-vlm",
            markdown="",
            error=f"vlm: all {len(pages_png)} page(s) failed transcription",
        )
    if failed_pages:
        notes.append(
            f"vlm: {len(failed_pages)} page(s) failed transcription: "
            f"{failed_pages}"
        )
        status = "partial"
    else:
        status = "processed"

    return ExtractionResult(
        status=status,
        extractor="pdf-vlm",
        markdown=markdown,
        assets=saved_assets,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_pages(src: Path) -> list[bytes]:
    """Render each PDF page to PNG bytes via pypdfium2."""
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    scale = float(os.environ.get("BRAIN_VLM_SCALE") or _DEFAULT_SCALE)
    out: list[bytes] = []
    pdf = pdfium.PdfDocument(str(src))
    try:
        for page in pdf:
            pil = page.render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            out.append(buf.getvalue())
    finally:
        pdf.close()
    return out


# ---------------------------------------------------------------------------
# Vision dispatch (anthropic / openai / gemini / local)
# ---------------------------------------------------------------------------

def _transcribe_page(
    *, png: bytes, provider: str, model: str, page_no: int
) -> str | None:
    try:
        if provider == "anthropic":
            return _vision_anthropic(png=png, model=model)
        if provider in ("openai", "local"):
            return _vision_openai_compatible(png=png, model=model, provider=provider)
        if provider == "gemini":
            return _vision_gemini(png=png, model=model)
    except Exception as exc:  # noqa: BLE001 — one page failing must not kill the doc
        _LOG.warning("vlm: %s/%s page %d failed (%r)", provider, model, page_no, exc)
        return None
    _LOG.warning("vlm: provider %r has no vision path", provider)
    return None


def _vision_anthropic(*, png: bytes, model: str) -> str | None:
    import anthropic

    client = anthropic.Anthropic()
    b64 = base64.standard_b64encode(png).decode()
    resp = client.messages.create(
        model=model,
        max_tokens=_MAX_OUTPUT_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    # Return the text (possibly "" for a genuinely blank page); None is
    # reserved for a real failure, which surfaces as an exception here.
    return "".join(parts)


def _vision_openai_compatible(*, png: bytes, model: str, provider: str) -> str | None:
    from openai import OpenAI

    kwargs: dict[str, object] = {}
    if provider == "local":
        base_url = os.environ.get("BRAIN_LOCAL_URL")
        if not base_url:
            _LOG.warning("vlm: local provider requires BRAIN_LOCAL_URL")
            return None
        kwargs["base_url"] = base_url
        kwargs["api_key"] = os.environ.get("BRAIN_LOCAL_API_KEY") or "not-needed"
    client = OpenAI(**kwargs)
    b64 = base64.standard_b64encode(png).decode()
    completion = client.chat.completions.create(
        model=model,
        max_completion_tokens=_MAX_OUTPUT_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
    )
    if not completion.choices:
        return None
    # "" is a blank page (success); None only on the structural failure above.
    return completion.choices[0].message.content or ""


def _vision_gemini(*, png: bytes, model: str) -> str | None:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _LOG.warning("vlm: gemini requires GOOGLE_API_KEY or GEMINI_API_KEY")
        return None
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=png, mime_type="image/png"),
            _PROMPT,
        ],
        config=types.GenerateContentConfig(max_output_tokens=_MAX_OUTPUT_TOKENS),
    )
    return getattr(resp, "text", None) or None
