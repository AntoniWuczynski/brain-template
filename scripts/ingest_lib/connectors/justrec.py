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

from .base import Snapshot, meeting_filename, normalize_attendees


def _to_snapshot(data: dict, rel_id: str) -> Snapshot | None:
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

    def pull(self, state) -> Iterator[Snapshot]:  # noqa: ANN001 - ConnectorState
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
                snap = _to_snapshot(data, f.relative_to(base).as_posix())
                if snap is not None:
                    yield snap
