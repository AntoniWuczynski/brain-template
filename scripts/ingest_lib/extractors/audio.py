"""Audio + subtitle/transcript extractor.

Two source classes, one honest contract:

- **Subtitles/transcripts** (``.vtt``, ``.srt``) parse deterministically into
  a timestamped Markdown transcript — no model, no system deps.
- **Audio** (``.m4a``, ``.mp3``, ...) is transcribed with a local Whisper
  model (``faster-whisper``, which bundles audio decoding via PyAV) when it
  is installed. When the ASR backend is absent the file is marked
  ``manual_review`` with the install command, never a hallucinated transcript
  (same honesty rule as the vision-LLM extractor's no-provider path).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Final, Literal

from .base import ExtractionResult

_LOG = logging.getLogger(__name__)

SUBTITLE_EXTENSIONS: Final[tuple[str, ...]] = (".vtt", ".srt")
AUDIO_EXTENSIONS: Final[tuple[str, ...]] = (
    ".m4a", ".mp3", ".wav", ".ogg", ".flac", ".m4b", ".aac",
)

_CUE_SEP = "-->"
_TIME_RE = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})")
# Strip only true markup (a letter or '/' right after '<'), so a literal
# "x < y > z" in the transcript is NOT eaten as a tag.
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_SEGMENT_SECONDS = 60                        # group cues into ~1-min paragraphs
# VTT non-cue blocks: skip so their bodies aren't mistaken for transcript.
_NONCUE_PREFIXES = ("WEBVTT", "NOTE", "STYLE", "REGION")


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    ext = src.suffix.lower()
    if ext in SUBTITLE_EXTENSIONS:
        return _extract_subtitles(src)
    if ext in AUDIO_EXTENSIONS:
        return _extract_audio(src)
    return ExtractionResult(
        status="manual_review", extractor="audio", markdown="",
        error=f"unrecognised audio/subtitle extension: {ext}",
    )


# --------------------------------------------------------------- subtitles

def _parse_time(s: str) -> float | None:
    m = _TIME_RE.search(s)
    if not m:
        return None
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000


def _parse_cues(text: str) -> tuple[list[tuple[float, str]], int]:
    """Return ``(cues, dropped)``: ``(start_seconds, text)`` per cue plus the
    number of blocks that looked like cues (had a ``-->`` line) but whose
    timing couldn't be parsed — so the caller can flag a partial extraction
    instead of silently claiming a complete transcript."""
    cues: list[tuple[float, str]] = []
    dropped = 0
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Skip VTT non-cue blocks (header, NOTE/STYLE/REGION) entirely.
        if lines[0].strip().split()[0].upper() in _NONCUE_PREFIXES:
            continue
        timing = next((i for i, ln in enumerate(lines) if _CUE_SEP in ln), None)
        if timing is None:
            continue
        start = _parse_time(lines[timing].split(_CUE_SEP)[0])
        if start is None:
            dropped += 1              # had a '-->' but an unparseable time
            continue
        cue = _TAG_RE.sub("", " ".join(lines[timing + 1:])).strip()
        if cue:
            cues.append((start, cue))
    return cues, dropped


def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _render_transcript(cues: list[tuple[float, str]]) -> str:
    """Group cues into ~1-minute paragraphs led by a ``[MM:SS]`` marker,
    collapsing consecutive duplicate lines (auto-caption rolling repeats)."""
    paras: list[str] = []
    seg_start: float | None = None
    buf: list[str] = []
    last: str | None = None
    for start, text in cues:
        if text == last:
            continue
        last = text
        if seg_start is None or start - seg_start >= _SEGMENT_SECONDS:
            if buf:
                paras.append(f"**[{_fmt_ts(seg_start or 0)}]** " + " ".join(buf))
                buf = []
            seg_start = start
        buf.append(text)
    if buf:
        paras.append(f"**[{_fmt_ts(seg_start or 0)}]** " + " ".join(buf))
    return "## Transcript\n\n" + "\n\n".join(paras) + "\n"


def _extract_subtitles(src: Path) -> ExtractionResult:
    try:
        raw = src.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return ExtractionResult(
            status="manual_review", extractor="audio-subtitle", markdown="",
            error=f"read failed: {exc}",
        )
    cues, dropped = _parse_cues(raw)
    if not cues:
        return ExtractionResult(
            status="manual_review", extractor="audio-subtitle", markdown="",
            error="no parseable subtitle cues",
        )
    notes = [f"audio-subtitle: {len(cues)} cue(s)"]
    # Some cues had a '-->' but an unparseable time: honest partial, not a
    # silent "complete" transcript.
    status: Literal["processed", "partial"] = "processed"
    if dropped:
        status = "partial"
        notes.append(f"audio-subtitle: {dropped} cue(s) dropped (unparseable timing)")
    return ExtractionResult(
        status=status, extractor="audio-subtitle",
        markdown=_render_transcript(cues), notes=notes,
    )


# ------------------------------------------------------------------- audio

def _extract_audio(src: Path) -> ExtractionResult:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError:
        return ExtractionResult(
            status="manual_review", extractor="audio-asr", markdown="",
            error="audio: no ASR backend — install faster-whisper to transcribe "
            "(uv pip install faster-whisper). It bundles audio decoding via "
            "PyAV; a system ffmpeg is only needed for exotic codecs.",
        )
    model_name = os.environ.get("BRAIN_WHISPER_MODEL", "base")
    try:
        model = WhisperModel(model_name)
        segments, _info = model.transcribe(str(src))
        cues = [(float(seg.start), seg.text.strip()) for seg in segments if seg.text.strip()]
    except Exception as exc:  # noqa: BLE001 — transcription can fail many ways (ffmpeg, model)
        return ExtractionResult(
            status="manual_review", extractor="audio-asr", markdown="",
            error=f"audio: transcription failed ({exc!r})",
        )
    if not cues:
        return ExtractionResult(
            status="partial", extractor="audio-asr",
            markdown="_(no speech detected)_\n",
            notes=[f"audio-asr: whisper {model_name}, no speech"],
        )
    return ExtractionResult(
        status="processed", extractor="audio-asr",
        markdown=_render_transcript(cues),
        notes=[f"audio-asr: whisper {model_name}, {len(cues)} segment(s)"],
    )
