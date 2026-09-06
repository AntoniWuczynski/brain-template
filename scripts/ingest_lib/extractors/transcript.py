"""Chat-transcript extractor (Claude Code sessions / claude.ai & ChatGPT
conversation exports).

All three producers normalise into ONE JSON snapshot schema (see
``connectors/_transcript_common.py``, ``connectors/claude_code.py`` and
``connectors/chat_export.py``), so this single extractor turns any of them
into a searchable note: title, date, source context, and the surviving
user/assistant prose exchange — with a processing-notes section recording
exactly what was dropped or redacted, so the loss is visible rather than
silent.

Registered by source-class prefix (``transcripts/claude_code/``,
``transcripts/chat_export/``) rather than by extension, since the snapshots
are ``.json`` — an extension ``text.py`` would otherwise claim.

Snapshot schema (all keys present; see ``_transcript_common.py`` for the
authoritative contract):

    {"connector": "claude_code"|"claude_ai"|"chatgpt", "id": "...",
     "title": "...", "date": "2026-09-03", "context": "...",
     "turns": [{"role": "user"|"assistant", "text": "..."}, ...],
     "stats": {...}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .base import ExtractionResult

_SPEAKER = {"user": "**User:**", "assistant": "**Assistant:**"}

_STAT_LABELS = {
    "dropped_thinking_blocks": "internal reasoning (\"thinking\") blocks dropped",
    "dropped_tool_uses": "tool calls dropped",
    "dropped_tool_results": "tool results dropped",
    "dropped_other_events": "non-conversation harness events dropped",
    "dropped_sidechain_lines": "subagent/sidechain lines dropped",
    "redacted_secrets": "likely secrets redacted",
    "truncated_long_turns": "oversized turns truncated",
    "dropped_non_text_parts": "non-text message parts dropped",
    "dropped_wrapper_turns": "harness wrapper turns dropped (no prose survived)",
    "dropped_overflow_turns": "turns dropped to stay under the per-session size cap",
}


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    try:
        raw = src.read_text(encoding="utf-8-sig", errors="replace")
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        return ExtractionResult(
            status="manual_review", extractor="transcript", markdown="",
            error=f"transcript: could not read snapshot ({exc})",
        )
    if not isinstance(data, dict):
        return ExtractionResult(
            status="manual_review", extractor="transcript", markdown="",
            error="transcript: snapshot is not a JSON object",
        )

    title = str(data.get("title") or "Untitled conversation").strip()
    date = str(data.get("date") or "").strip()
    connector = str(data.get("connector") or "").strip()
    context = str(data.get("context") or "").strip()
    turns_raw = data.get("turns")
    turns = turns_raw if isinstance(turns_raw, list) else []
    stats_raw = data.get("stats")
    stats = stats_raw if isinstance(stats_raw, dict) else {}

    lines = [f"# {title}", ""]
    meta = []
    if date:
        meta.append(f"**Date:** {date}")
    if connector:
        meta.append(f"**Source:** {connector}")
    if context:
        meta.append(f"**Context:** {context}")
    if meta:
        lines += ["  \n".join(meta), ""]

    lines += ["## Transcript", ""]
    kept = 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if role == "marker":
            # A synthetic gap-elision marker from cap_session_turns (review
            # N5) — rendered inline with no speaker label, and not counted
            # towards "kept" (it isn't a turn either party spoke).
            lines += ["", f"_{text.strip()}_", ""]
            continue
        speaker = _SPEAKER.get(role) if isinstance(role, str) else None
        if speaker is None:
            continue
        lines += [speaker, "", text.strip(), ""]
        kept += 1

    status: Literal["processed", "partial", "manual_review"]
    if kept == 0:
        status = "partial"
        lines.append("_(no transcript content survived extraction)_")
    else:
        status = "processed"

    notes = [f"transcript: {connector or 'unknown'} snapshot, {kept} turn(s) kept"]
    dropped_lines = []
    for key, label in _STAT_LABELS.items():
        val = stats.get(key)
        if isinstance(val, int) and val:
            dropped_lines.append(f"{val} {label}")
            notes.append(f"transcript: {val} {label}")

    lines += ["## Processing notes", ""]
    lines.append(
        "Only prose turns from the user and the assistant are kept above — "
        "tool calls, tool results, and internal reasoning are extracted out."
    )
    if dropped_lines:
        lines += ["", *[f"- {d}" for d in dropped_lines]]
    lines.append("")

    return ExtractionResult(
        status=status, extractor="transcript", markdown="\n".join(lines), notes=notes,
    )
