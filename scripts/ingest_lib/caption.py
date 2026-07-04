"""Vision captions for extracted figures and tables.

MinerU pulls figures/tables out of PDFs into ``*_assets/<sha>.<ext>`` and
leaves bare ``![](…)`` links in the processed Markdown. Nothing described them,
so a figure was invisible to search and contributed nothing to summaries. This
module captions each image with a vision-capable model and writes the caption
*inline* beneath the image, so it (a) shows up in Obsidian and (b) gets embedded
on the next ``--rebuild-search-index`` — making figures searchable.

Design, consistent with the other enrichers:

- A durable cache (``metadata/captions.jsonl``, keyed by image content hash)
  means a caption is paid for once even across re-extraction.
- Inline insertion carries a ``<!-- caption: <hash> -->`` marker, so re-runs are
  idempotent and a caption can be replaced in place.
- Tiny images (< a few KB) are skipped as decorative noise. ``--limit`` bounds
  the number of *new* vision calls per run; vision is never auto-run on ingest.

Pure pieces (ref finding, hashing, the inline upsert) are unit-tested; the
vision call + file walk are verified by running.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import VaultPaths
from .notes import _atomic_write
from .summarize import _select_model, _select_provider, is_enabled

_IMG_RE = re.compile(
    r"!\[\]\(([^)\s]+?_assets/[^)\s]+?\.(?:jpe?g|png|gif|webp))\)", re.IGNORECASE
)
_MIN_BYTES = 3000          # skip decorative/noise fragments
_HASH_LEN = 16
_MAX_CAPTION_TOKENS = 200

_CAPTION_PROMPT = (
    "This image is a figure, chart, diagram, or table extracted from a "
    "document in someone's study/work notes. In one to three sentences, "
    "describe what it shows and its key takeaway, so it can be found by "
    "search. Be concrete and factual; do not guess at text you cannot read. "
    "Reply with the description only — no preamble."
)

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


# Providers this module actually has a vision path for. gemini is a valid
# summarizer provider but has no caption path here, so guard against walking
# the whole archive emitting one warning per figure.
_VISION_PROVIDERS = frozenset({"anthropic", "openai", "local"})


@dataclass(frozen=True)
class CaptionStats:
    captioned: int = 0          # new vision calls
    cached: int = 0             # reused from the sidecar
    skipped_small: int = 0
    skipped_missing: int = 0
    no_llm: bool = False
    no_vision_provider: bool = False


# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------

def image_refs(md_text: str) -> list[str]:
    """Relative paths of every extracted-asset image referenced in ``md_text``."""
    return _IMG_RE.findall(md_text)


def image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upsert_caption(md_text: str, ref: str, image_hash: str, caption: str) -> str:
    """Insert/replace a one-line caption right after ``![](ref)``.

    Idempotent: keyed on ``image_hash`` via a marker comment, so re-running with
    the same caption is a no-op and a changed caption replaces in place.
    """
    full = f"![]({ref})"
    marker = f"<!-- caption: {image_hash} -->"
    strip_re = re.compile(r"\n\n" + re.escape(marker) + r"\n_Figure:[^\n]*")
    md_text = strip_re.sub("", md_text)
    idx = md_text.find(full)
    if idx < 0:
        return md_text
    end = idx + len(full)
    one_line = " ".join(caption.split())
    return md_text[:end] + f"\n\n{marker}\n_Figure: {one_line}_" + md_text[end:]


# ---------------------------------------------------------------------------
# vision glue (verified by running)
# ---------------------------------------------------------------------------

def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")


def caption_image(path: Path, *, logger: logging.Logger | None = None) -> str | None:
    """Caption one image via the configured (vision-capable) provider."""
    log = logger or logging.getLogger(__name__)
    provider = _select_provider()
    if provider is None:
        return None
    model = _select_model(provider)
    try:
        data = path.read_bytes()
    except OSError as exc:
        log.warning("caption: cannot read %s (%s)", path.name, exc)
        return None
    b64 = base64.standard_b64encode(data).decode()
    media_type = _media_type(path)
    if provider == "anthropic":
        return _caption_anthropic(model, b64, media_type, log)
    if provider in ("openai", "local"):
        return _caption_openai_compatible(provider, model, b64, media_type, log)
    log.warning("caption: provider %r has no vision path here — skipping", provider)
    return None


def _caption_anthropic(model: str, b64: str, media_type: str, log: logging.Logger) -> str | None:
    try:
        import anthropic
    except ImportError as exc:
        log.warning("caption: anthropic SDK missing (%s)", exc)
        return None
    try:
        # Construct INSIDE the try: a missing key raises at .create() time
        # (TypeError: could not resolve authentication) which anthropic.APIError
        # does not catch — that would crash the whole caption run.
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=_MAX_CAPTION_TOKENS,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": media_type, "data": b64}},
                {"type": "text", "text": _CAPTION_PROMPT},
            ]}],
        )
    except Exception as exc:  # noqa: BLE001 — SDK error hierarchy varies; degrade to skip
        log.warning("caption: anthropic call failed (%r)", exc)
        return None
    text = " ".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    return text or None


def _caption_openai_compatible(
    provider: str, model: str, b64: str, media_type: str, log: logging.Logger
) -> str | None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        log.warning("caption: openai SDK missing (%s) — needed for %s", exc, provider)
        return None
    kwargs: dict[str, object] = {}
    if provider == "local":
        base = os.environ.get("BRAIN_LOCAL_URL")
        if not base:
            log.warning("caption: local provider requires BRAIN_LOCAL_URL")
            return None
        kwargs["base_url"] = base
        kwargs["api_key"] = os.environ.get("BRAIN_LOCAL_API_KEY") or "not-needed"
    data_uri = f"data:{media_type};base64,{b64}"
    try:
        # Construct inside the try: OpenAI(**kwargs) raises OpenAIError at
        # construction when no key is set, outside any handler otherwise.
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            # max_completion_tokens, not max_tokens: the default openai model
            # (gpt-5-mini, a reasoning model) rejects the deprecated param,
            # which broke captioning entirely for the openai provider.
            max_completion_tokens=_MAX_CAPTION_TOKENS,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
        )
    except Exception as exc:  # noqa: BLE001 — SDK error hierarchy varies
        log.warning("caption: %s call failed (%r)", provider, exc)
        return None
    if not resp.choices:
        return None
    return (resp.choices[0].message.content or "").strip() or None


# ---------------------------------------------------------------------------
# caption cache (durable sidecar, gitignored)
# ---------------------------------------------------------------------------

def _load_captions(paths: VaultPaths) -> dict[str, str]:
    path = paths.metadata / "captions.jsonl"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "hash" in row and "caption" in row:
            out[row["hash"]] = row["caption"]
    return out


def _append_caption(paths: VaultPaths, image_hash: str, caption: str, model: str) -> None:
    path = paths.metadata / "captions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"hash": image_hash, "caption": caption, "model": model},
        ensure_ascii=False, sort_keys=True,
    ) + "\n"
    if not path.exists():
        fd, tmp = tempfile.mkstemp(prefix=".captions-", suffix=".jsonl", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    else:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def rebuild_captions(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
    limit: int | None = None,
    force: bool = False,
    min_bytes: int = _MIN_BYTES,
) -> CaptionStats:
    """Caption extracted figures across ``archive/processed/``.

    ``limit`` bounds the number of *new* vision calls (cost control); cache hits
    don't count. ``force`` re-captions even when cached/inlined.
    """
    if not is_enabled():
        logger.warning("caption: no LLM provider configured — skipping")
        return CaptionStats(no_llm=True)

    provider = _select_provider()
    if provider not in _VISION_PROVIDERS:
        # Fail once, up front, instead of one warning per figure across the
        # whole archive (and never reaching the --limit early exit).
        logger.warning(
            "caption: provider %r has no vision path here (need one of %s) — skipping",
            provider, ", ".join(sorted(_VISION_PROVIDERS)),
        )
        return CaptionStats(no_vision_provider=True)

    model = f"{provider}/{_select_model(provider)}"
    cache = _load_captions(paths)
    captioned = cached = skipped_small = skipped_missing = 0

    for md_path in sorted(paths.archive_processed.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        # Dedupe refs: a diagram reused across slides appears more than once,
        # but captioning cost is per unique image and upsert targets the first
        # occurrence anyway.
        refs = list(dict.fromkeys(image_refs(text)))
        if not refs:
            continue
        changed = False
        for ref in refs:
            if limit is not None and captioned >= limit:
                break
            img_path = (md_path.parent / ref).resolve()
            if not img_path.is_file():
                skipped_missing += 1
                continue
            data = img_path.read_bytes()
            if len(data) < min_bytes:
                skipped_small += 1
                continue
            key = image_sha256(data)[:_HASH_LEN]
            marker = f"<!-- caption: {key} -->"
            if not force and marker in text:
                continue  # already inlined — idempotent skip
            if not force and key in cache:
                caption = cache[key]
                cached += 1
            else:
                caption = caption_image(img_path, logger=logger) or ""
                if not caption:
                    continue
                _append_caption(paths, key, caption, model)
                cache[key] = caption
                captioned += 1
                logger.info("caption: %s (%s)", img_path.name[:16], md_path.name)
            text = upsert_caption(text, ref, key, caption)
            changed = True
        if changed:
            _atomic_write(md_path, text)
        if limit is not None and captioned >= limit:
            break
    return CaptionStats(
        captioned=captioned, cached=cached,
        skipped_small=skipped_small, skipped_missing=skipped_missing,
    )
