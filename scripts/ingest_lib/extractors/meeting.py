"""Meeting-snapshot extractor (Granola / justREC, one shape).

Both meeting connectors normalise their source into ONE JSON snapshot schema
(see ``connectors/granola.py`` and ``connectors/justrec.py``), so this single
extractor turns either into a searchable meeting note: title, date, attendees
linked to ``knowledge/people/``, the AI summary, and the transcript.

Registered by source-class prefix (``meetings/granola/`` ,
``meetings/justrec/``) rather than by extension, since the snapshots are
``.json`` — an extension ``text.py`` would otherwise claim.

Snapshot schema (all keys optional except a title or some content):

    {"connector": "granola"|"justrec", "id": "...", "title": "...",
     "date": "2026-07-12", "attendees": ["Alice Smith", ...],
     "summary": "...", "transcript": "..."}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from .base import ExtractionResult

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("-", name.strip().lower()).strip("-")


def _attendee_links(attendees: list[str]) -> str:
    items: list[str] = []
    for a in attendees:
        if not isinstance(a, str) or not a.strip():
            continue
        slug = _slug(a)
        # A name that slugs to empty (e.g. fully non-Latin) can't form a valid
        # node id — render it as plain text rather than a broken [[.../]] link.
        items.append(f"[[knowledge/people/{slug}]]" if slug else a.strip())
    return ", ".join(items) if items else "_(none recorded)_"


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    try:
        raw = src.read_text(encoding="utf-8-sig", errors="replace")
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        return ExtractionResult(
            status="manual_review", extractor="meeting", markdown="",
            error=f"meeting: could not read snapshot ({exc})",
        )
    if not isinstance(data, dict):
        return ExtractionResult(
            status="manual_review", extractor="meeting", markdown="",
            error="meeting: snapshot is not a JSON object",
        )

    title = str(data.get("title") or "").strip() or "Untitled meeting"
    date = str(data.get("date") or "").strip()
    attendees = data.get("attendees")
    attendees = [a for a in attendees if isinstance(a, str)] if isinstance(attendees, list) else []
    summary = str(data.get("summary") or "").strip()
    transcript = str(data.get("transcript") or "").strip()
    connector = str(data.get("connector") or "").strip()

    status: Literal["processed", "partial", "manual_review"]
    if not summary and not transcript:
        # Nothing but metadata: honest partial, not an invented body.
        status = "partial"
    else:
        status = "processed"

    lines = [f"# {title}", ""]
    meta = []
    if date:
        meta.append(f"**Date:** {date}")
    meta.append(f"**Attendees:** {_attendee_links(attendees)}")
    if connector:
        meta.append(f"**Source:** {connector}")
    lines += ["  \n".join(meta), ""]
    lines += ["## Summary", "", summary or "_(no summary)_", ""]
    lines += ["## Transcript", "", transcript or "_(no transcript captured)_", ""]

    notes = [f"meeting: {connector or 'unknown'} snapshot, {len(attendees)} attendee(s)"]
    if status == "partial":
        notes.append("meeting: no summary or transcript in snapshot")
    return ExtractionResult(
        status=status, extractor="meeting",
        markdown="\n".join(lines), notes=notes,
    )
