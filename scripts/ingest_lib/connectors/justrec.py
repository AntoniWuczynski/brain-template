"""justREC meeting connector (local-first).

justREC (justrec.site) is a macOS recorder that writes every meeting — audio,
transcript, summary — as plain files to a folder you choose. This connector
reads that folder's JSON exports and normalises each into the common meeting
snapshot schema (see ``extractors/meeting.py``) written to
``inbox/meetings/justrec/``. No API, no auth: the "fetch" is a local file
read, so it works offline and needs no network.

Point it at the folder with ``BRAIN_JUSTREC_DIR``; without it (or if the
folder is absent) ``pull`` yields nothing.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from .base import Snapshot, meeting_filename, normalize_attendees

if TYPE_CHECKING:
    from .state import ConnectorState


class _Meeting(TypedDict, total=False):
    """The fields this connector reads from one justREC export's JSON.
    ``total=False``: a real export may omit any of these, and every read in
    ``_to_snapshot`` already tolerates absence via ``.get(...)``."""
    id: str
    title: str
    date: str
    createdAt: str
    created_at: str
    attendees: object
    participants: object
    summary: str
    notes: str
    transcript: str
    text: str


def _meeting_from_json(obj: object) -> _Meeting:
    """Narrow one decoded justREC export object to a ``_Meeting``.

    A wrong-typed field is simply omitted rather than raising;
    ``_to_snapshot``'s ``.get(...)`` calls already tolerate absence.
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
    created_at_camel = obj.get("createdAt")
    if isinstance(created_at_camel, str):
        out["createdAt"] = created_at_camel
    created_at_snake = obj.get("created_at")
    if isinstance(created_at_snake, str):
        out["created_at"] = created_at_snake
    summary = obj.get("summary")
    if isinstance(summary, str):
        out["summary"] = summary
    notes = obj.get("notes")
    if isinstance(notes, str):
        out["notes"] = notes
    transcript = obj.get("transcript")
    if isinstance(transcript, str):
        out["transcript"] = transcript
    text = obj.get("text")
    if isinstance(text, str):
        out["text"] = text
    if "attendees" in obj:
        out["attendees"] = obj["attendees"]
    if "participants" in obj:
        out["participants"] = obj["participants"]
    return out


def _to_snapshot(data: _Meeting, rel_id: str) -> Snapshot | None:
    # native id: an explicit id, else the file's path relative to the export
    # dir — unique per file (the bare stem collides across subfolders under
    # rglob and would re-snapshot forever).
    mid = str(data.get("id") or rel_id).strip()
    if not mid:
        return None
    title = str(data.get("title") or Path(rel_id).stem or "Untitled meeting").strip()
    date = str(data.get("date") or data.get("createdAt") or data.get("created_at") or "")[:10]
    normalized = {
        "connector": "justrec",
        "id": mid,
        "title": title,
        "date": date,
        "attendees": normalize_attendees(
            data.get("attendees") or data.get("participants")
        ),
        "summary": str(data.get("summary") or data.get("notes") or "").strip(),
        "transcript": str(data.get("transcript") or data.get("text") or "").strip(),
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Snapshot(
        source_class="meetings/justrec", native_id=mid,
        filename=meeting_filename(date, title, mid), payload=payload,
    )


class JustrecConnector:
    name = "justrec"

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]:
        folder = os.environ.get("BRAIN_JUSTREC_DIR")
        if not folder:
            return
        base = Path(folder).expanduser()
        if not base.is_dir():
            return
        for f in sorted(base.rglob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8-sig", errors="replace"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                meeting = _meeting_from_json(data)
                snap = _to_snapshot(meeting, f.relative_to(base).as_posix())
                if snap is not None:
                    yield snap
