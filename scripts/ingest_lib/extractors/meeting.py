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

:func:`load_snapshot` is the schema read, kept apart from the rendering
below because a second consumer needs it: :mod:`ingest_lib.meetings`
promotes an archived snapshot into a first-class
``knowledge/meetings/<YYYY>/`` note, and re-parsing the same JSON there
would be two spellings of one schema, free to drift. Both therefore see the
same title, date and attendee list — including
``connectors.base.normalize_attendees``, so a snapshot that carries
``[{"name": "Alice Smith"}]`` (the shape the connectors normalise FROM)
reads identically here, and an attendee name is rendered inert
(``connectors.base.safe_display_name``) on the way out of the archive just as
it was on the way in.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..connectors.base import normalize_attendees, safe_display_name
from .base import ExtractionResult

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def attendee_slug(name: str) -> str:
    """``people/`` slug for an attendee display name.

    Accents are folded to their base letter first (``Lluís Masanes`` ->
    ``lluis-masanes``, ``Antoni Wuczyński`` -> ``antoni-wuczynski``) rather
    than being punched out as separators: the live vault's person notes are
    slugged the folded way, so the unfolded spelling (``llu-s-masanes``)
    links to — and resolves to — nothing. A name that folds to nothing at
    all (fully non-Latin) yields ``""``; callers must handle that rather
    than emit an empty node id.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SLUG_RE.sub("-", folded.strip().lower()).strip("-")


@dataclass(frozen=True)
class MeetingSnapshot:
    """One connector snapshot, parsed. Every field is already trimmed, and
    absent/ill-typed keys read as empty rather than raising — the snapshot
    is archived bytes, so a re-read of it must never fail differently than
    the ingest that accepted it."""

    connector: str
    native_id: str
    title: str
    date: str
    attendees: tuple[str, ...]
    summary: str
    transcript: str


def _text(raw: object) -> str:
    return raw.strip() if isinstance(raw, str) else ""


def load_snapshot(src: Path) -> tuple[MeetingSnapshot | None, str]:
    """Parse a snapshot file into ``(snapshot, error)``.

    Exactly one of the two is set. The error is returned rather than raised
    so both callers (the extractor's ``manual_review`` path and the
    promotion pass's skip report) can record it verbatim.
    """
    try:
        raw = src.read_text(encoding="utf-8-sig", errors="replace")
        parsed: object = json.loads(raw)
    except (OSError, ValueError) as exc:
        return None, f"meeting: could not read snapshot ({exc})"
    if not isinstance(parsed, dict):
        return None, "meeting: snapshot is not a JSON object"
    # JSON object keys are always strings; the annotation pins the value
    # type so nothing downstream reads an untyped mapping.
    data: dict[str, object] = {str(key): value for key, value in parsed.items()}
    return MeetingSnapshot(
        connector=_text(data.get("connector")),
        native_id=_text(data.get("id")),
        title=_text(data.get("title")) or "Untitled meeting",
        date=_text(data.get("date")),
        attendees=tuple(normalize_attendees(data.get("attendees"))),
        summary=_text(data.get("summary")),
        transcript=_text(data.get("transcript")),
    ), ""


def _attendee_links(attendees: tuple[str, ...]) -> str:
    items: list[str] = []
    for a in attendees:
        slug = attendee_slug(a)
        # A name that slugs to empty (e.g. fully non-Latin) can't form a valid
        # node id — render it as plain text rather than a broken [[.../]] link.
        items.append(f"[[knowledge/people/{slug}]]" if slug else a)
    return ", ".join(items) if items else "_(none recorded)_"


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    snapshot, error = load_snapshot(src)
    if snapshot is None:
        return ExtractionResult(
            status="manual_review", extractor="meeting", markdown="", error=error,
        )

    status: Literal["processed", "partial", "manual_review"]
    if not snapshot.summary and not snapshot.transcript:
        # Nothing but metadata: honest partial, not an invented body.
        status = "partial"
    else:
        status = "processed"

    # The title is connector-supplied text going into a Markdown heading, so
    # it goes through the same one-line sanitiser the attendee names do (a
    # title carrying newlines and a ``[[...]]`` link would otherwise open its
    # own sections in this note). ``MeetingSnapshot.title`` keeps the raw
    # value: ``ingest_lib.meetings`` needs to SEE the control characters to
    # refuse the snapshot rather than name a node from it.
    lines = [f"# {safe_display_name(snapshot.title)}", ""]
    meta = []
    if snapshot.date:
        meta.append(f"**Date:** {snapshot.date}")
    meta.append(f"**Attendees:** {_attendee_links(snapshot.attendees)}")
    if snapshot.connector:
        meta.append(f"**Source:** {snapshot.connector}")
    lines += ["  \n".join(meta), ""]
    lines += ["## Summary", "", snapshot.summary or "_(no summary)_", ""]
    lines += ["## Transcript", "", snapshot.transcript or "_(no transcript captured)_", ""]

    notes = [
        f"meeting: {snapshot.connector or 'unknown'} snapshot, "
        f"{len(snapshot.attendees)} attendee(s)"
    ]
    if status == "partial":
        notes.append("meeting: no summary or transcript in snapshot")
    return ExtractionResult(
        status=status, extractor="meeting",
        markdown="\n".join(lines), notes=notes,
    )
