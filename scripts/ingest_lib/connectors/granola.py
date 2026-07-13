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

from .base import Snapshot, meeting_filename, normalize_attendees

_LOG = logging.getLogger(__name__)
_API_URL = "https://api.granola.ai/v1/meetings"


def _fetch_meetings(api_key: str) -> list[dict]:
    """GET the meetings list. Isolated so tests can stub it (no network)."""
    url = os.environ.get("GRANOLA_API_URL", _API_URL)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https API
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, dict):
        meetings = data.get("meetings")
        return meetings if isinstance(meetings, list) else []
    return data if isinstance(data, list) else []


def _to_snapshot(meeting: dict) -> Snapshot | None:
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

    def pull(self, state) -> Iterator[Snapshot]:  # noqa: ANN001 - ConnectorState
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
