"""Granola meeting connector.

Pulls notes from Granola's public API (https://docs.granola.ai/introduction,
shape confirmed against a live pull 2026-09-17) and normalises each into the
common meeting snapshot schema (see ``extractors/meeting.py``) written to
``inbox/meetings/granola/``. The network fetch is the only non-deterministic
step and happens before the archive boundary, so downstream ingest stays
deterministic and idempotent.

``GET /v1/notes`` lists note ids (cursor-paginated via ``hasMore``/``cursor``);
``GET /v1/notes/{id}?include=transcript`` carries the attendees, summary and
transcript. Requires a Business/Enterprise plan key (``grn_...``).

Auth: ``GRANOLA_API_KEY`` from the environment (never a flag). Without it,
``pull`` yields nothing.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import TYPE_CHECKING, TypedDict

from .base import Snapshot, meeting_filename, normalize_attendees

if TYPE_CHECKING:
    from .state import ConnectorState

_LOG = logging.getLogger(__name__)
_API_URL = "https://public-api.granola.ai/v1"
# The API allows 5 requests/second sustained; one detail GET per note.
_REQUEST_GAP_S = 0.25


class _Meeting(TypedDict, total=False):
    """The fields this connector reads from one Granola note object."""
    id: str
    title: str
    created_at: str
    attendees: list[dict[str, str]]
    summary: str
    transcript: str


def _get_json(url: str, api_key: str) -> dict[str, object]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https API
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("granola: expected a JSON object")
    return {str(k): v for k, v in data.items()}


def _str(obj: dict[str, object], key: str) -> str:
    value = obj.get(key)
    return value if isinstance(value, str) else ""


def _render_transcript(raw: object) -> str:
    """``[{speaker: {source, diarization_label}, text}, ...]`` → one
    ``Speaker: text`` line per segment."""
    if not isinstance(raw, list):
        return ""
    lines: list[str] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        speaker = seg.get("speaker")
        label = ""
        if isinstance(speaker, dict):
            for key in ("diarization_label", "source"):
                value = speaker.get(key)
                if isinstance(value, str) and value:
                    label = value
                    break
        lines.append(f"{label}: {text.strip()}" if label else text.strip())
    return "\n".join(lines)


def _meeting_from_json(obj: dict[str, object]) -> _Meeting:
    """Narrow one ``GET /notes/{id}`` body to a ``_Meeting``."""
    attendees: list[dict[str, str]] = []
    raw_attendees = obj.get("attendees")
    if isinstance(raw_attendees, list):
        for a in raw_attendees:
            if isinstance(a, dict) and isinstance(a.get("name"), str):
                attendees.append({"name": a["name"]})
    summary = next(
        (v for k in ("summary_markdown", "summary_text",
                     "private_notes_markdown", "private_notes_text")
         if (v := _str(obj, k).strip())),
        "",
    )
    return {
        "id": _str(obj, "id"),
        "title": _str(obj, "title"),
        "created_at": _str(obj, "created_at"),
        "attendees": attendees,
        "summary": summary,
        "transcript": _render_transcript(obj.get("transcript")),
    }


def _list_note_ids(api_key: str, base: str) -> list[str]:
    ids: list[str] = []
    cursor = ""
    while True:
        url = f"{base}/notes" + (f"?{urllib.parse.urlencode({'cursor': cursor})}" if cursor else "")
        page = _get_json(url, api_key)
        notes = page.get("notes")
        if isinstance(notes, list):
            ids.extend(n["id"] for n in notes if isinstance(n, dict) and isinstance(n.get("id"), str))
        next_cursor = page.get("cursor")
        if page.get("hasMore") is not True or not isinstance(next_cursor, str) or not next_cursor:
            return ids
        cursor = next_cursor
        time.sleep(_REQUEST_GAP_S)


def _fetch_meetings(api_key: str) -> list[_Meeting]:
    """List every note, then GET each with its transcript. Isolated so tests
    can stub it (no network)."""
    base = os.environ.get("GRANOLA_API_URL", _API_URL).rstrip("/")
    meetings: list[_Meeting] = []
    for note_id in _list_note_ids(api_key, base):
        time.sleep(_REQUEST_GAP_S)
        quoted = urllib.parse.quote(note_id, safe="")
        meetings.append(_meeting_from_json(
            _get_json(f"{base}/notes/{quoted}?include=transcript", api_key)
        ))
    return meetings


def _to_snapshot(meeting: _Meeting) -> Snapshot | None:
    mid = str(meeting.get("id") or "").strip()
    if not mid:
        return None
    title = str(meeting.get("title") or "Untitled meeting").strip()
    date = str(meeting.get("created_at") or "")[:10]
    normalized = {
        "connector": "granola",
        "id": mid,
        "title": title,
        "date": date,
        "attendees": normalize_attendees(meeting.get("attendees")),
        "summary": str(meeting.get("summary") or "").strip(),
        "transcript": str(meeting.get("transcript") or "").strip(),
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Snapshot(
        source_class="meetings/granola", native_id=mid,
        filename=meeting_filename(date, title, mid), payload=payload,
    )


class GranolaConnector:
    name = "granola"

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]:
        api_key = os.environ.get("GRANOLA_API_KEY")
        if not api_key:
            return
        try:
            meetings = _fetch_meetings(api_key)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            # The fetch is the one networked step: a transient API/network
            # error must degrade to "nothing pulled this run", never crash.
            _LOG.warning("granola: fetch failed (%r) — pulling nothing this run", exc)
            return
        for meeting in meetings:
            snap = _to_snapshot(meeting)
            if snap is not None:
                yield snap
