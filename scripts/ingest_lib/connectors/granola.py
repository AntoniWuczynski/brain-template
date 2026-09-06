"""Granola meeting connector.

Pulls meetings from Granola's API and normalises each into the common meeting
snapshot schema (see ``extractors/meeting.py``) written to
``inbox/meetings/granola/``. The network fetch is the only non-deterministic
step and happens before the archive boundary, so downstream ingest stays
deterministic and idempotent.

Auth: ``GRANOLA_API_KEY`` from the environment (never a flag). Without it,
``pull`` yields nothing. NOTE: the exact API response shape should be
confirmed against a real pull — the field mapping below is defensive and
tolerant, but Granola may name fields differently.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import TYPE_CHECKING, TypedDict

from .base import Snapshot, meeting_filename, normalize_attendees

if TYPE_CHECKING:
    from .state import ConnectorState

_LOG = logging.getLogger(__name__)
_API_URL = "https://api.granola.ai/v1/meetings"


class _Meeting(TypedDict, total=False):
    """The fields this connector reads from one Granola API meeting object.
    ``total=False``: the API may omit any of these (see the module docstring
    — the exact response shape is unconfirmed), and every read below already
    tolerates absence via ``.get(...)``."""
    id: str
    title: str
    date: str
    created_at: str
    attendees: object
    participants: object
    summary: str
    notes: str
    transcript: str


def _meeting_from_json(obj: object) -> _Meeting:
    """Narrow one decoded Granola meeting object to a ``_Meeting``.

    Every field is optional (``total=False``) — the exact API shape is
    unconfirmed (see module docstring) — so a wrong-typed field is simply
    omitted rather than raising; ``_to_snapshot``'s ``.get(...)`` calls
    already tolerate absence.
    """
    if not isinstance(obj, dict):
        return {}
    out: _Meeting = {}
    id_ = obj.get("id")
    if isinstance(id_, str):
        out["id"] = id_
    title = obj.get("title")
    if isinstance(title, str):
        out["title"] = title
    date = obj.get("date")
    if isinstance(date, str):
        out["date"] = date
    created_at = obj.get("created_at")
    if isinstance(created_at, str):
        out["created_at"] = created_at
    summary = obj.get("summary")
    if isinstance(summary, str):
        out["summary"] = summary
    notes = obj.get("notes")
    if isinstance(notes, str):
        out["notes"] = notes
    transcript = obj.get("transcript")
    if isinstance(transcript, str):
        out["transcript"] = transcript
    if "attendees" in obj:
        out["attendees"] = obj["attendees"]
    if "participants" in obj:
        out["participants"] = obj["participants"]
    return out


def _fetch_meetings(api_key: str) -> list[_Meeting]:
    """GET the meetings list. Isolated so tests can stub it (no network)."""
    url = os.environ.get("GRANOLA_API_URL", _API_URL)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https API
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, dict):
        meetings = data.get("meetings")
        if isinstance(meetings, list):
            return [_meeting_from_json(m) for m in meetings if isinstance(m, dict)]
        return []
    if isinstance(data, list):
        return [_meeting_from_json(m) for m in data if isinstance(m, dict)]
    return []


def _to_snapshot(meeting: _Meeting) -> Snapshot | None:
    mid = str(meeting.get("id") or "").strip()
    if not mid:
        return None
    title = str(meeting.get("title") or "Untitled meeting").strip()
    date = str(meeting.get("date") or meeting.get("created_at") or "")[:10]
    normalized = {
        "connector": "granola",
        "id": mid,
        "title": title,
        "date": date,
        "attendees": normalize_attendees(
            meeting.get("attendees") or meeting.get("participants")
        ),
        "summary": str(meeting.get("summary") or meeting.get("notes") or "").strip(),
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
            if isinstance(meeting, dict):
                snap = _to_snapshot(meeting)
                if snap is not None:
                    yield snap
