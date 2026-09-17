"""Notion note-page extractor (``notes/notion/``).

Every Notion page the connector pulls that does NOT carry an AI
meeting-notes block (see ``connectors/notion.py``) lands here as one JSON
snapshot per content version:

    {"connector": "notion", "id": "...", "title": "...", "url": "...",
     "created_time": "...", "markdown": "...",
     "truncated": bool, "unknown_blocks": int}

``last_edited_time`` is deliberately NOT in this schema: it lives only in
the snapshot's FILENAME (derived from the page's listing metadata, not the
payload — see ``connectors/notion.py``'s module docstring), so a
timestamp-only edit (a property change, a page move) never changes the
payload bytes with no real content change.

Registered by source-class prefix (``notes/notion/``) rather than by
extension, since the snapshot is ``.json`` — an extension ``text.py``
would otherwise claim.

:func:`load_snapshot` is the schema read, kept apart from the rendering
below the same way ``extractors/meeting.py`` splits its own snapshot read
from its rendering, in case a second consumer ever needs the parsed shape
without the Markdown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..connectors.base import safe_display_name
from .base import ExtractionResult


@dataclass(frozen=True)
class NotionPageSnapshot:
    """One connector snapshot, parsed. Absent/ill-typed keys read as empty
    rather than raising — the snapshot is archived bytes, so a re-read of
    it must never fail differently than the ingest that accepted it."""

    native_id: str
    title: str
    url: str
    created_time: str
    markdown: str
    truncated: bool
    unknown_blocks: int


def _text(raw: object) -> str:
    return raw.strip() if isinstance(raw, str) else ""


def _markdown_field(raw: object) -> str:
    # Not stripped like the metadata fields above: the payload's own
    # markdown is rendered verbatim under "## Content" below, so this
    # keeps whatever leading/trailing whitespace the source markdown had.
    return raw if isinstance(raw, str) else ""


def load_snapshot(src: Path) -> tuple[NotionPageSnapshot | None, str]:
    """Parse a snapshot file into ``(snapshot, error)``.

    Exactly one of the two is set, mirroring
    ``extractors.meeting.load_snapshot`` — the error is returned rather
    than raised so the extractor's ``manual_review`` path can record it.
    """
    try:
        raw = src.read_text(encoding="utf-8-sig", errors="replace")
        parsed: object = json.loads(raw)
    except (OSError, ValueError) as exc:
        return None, f"notion_page: could not read snapshot ({exc})"
    if not isinstance(parsed, dict):
        return None, "notion_page: snapshot is not a JSON object"
    # JSON object keys are always strings; the annotation pins the value
    # type so nothing downstream reads an untyped mapping.
    data: dict[str, object] = {str(key): value for key, value in parsed.items()}
    unknown_blocks = data.get("unknown_blocks")
    return NotionPageSnapshot(
        native_id=_text(data.get("id")),
        title=_text(data.get("title")) or "Untitled page",
        url=_text(data.get("url")),
        created_time=_text(data.get("created_time")),
        markdown=_markdown_field(data.get("markdown")),
        truncated=bool(data.get("truncated")),
        unknown_blocks=unknown_blocks if isinstance(unknown_blocks, int) else 0,
    ), ""


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    snapshot, error = load_snapshot(src)
    if snapshot is None:
        return ExtractionResult(
            status="manual_review", extractor="notion_page", markdown="", error=error,
        )

    status: Literal["processed", "partial", "manual_review"]
    notes: list[str] = []
    has_content = bool(snapshot.markdown.strip())
    if not has_content or snapshot.truncated:
        status = "partial"
        if not has_content:
            notes.append("notion_page: no content in snapshot")
        if snapshot.truncated:
            notes.append("notion_page: Notion markdown export was truncated (~20k block limit)")
    else:
        status = "processed"
    if snapshot.unknown_blocks:
        notes.append(f"notion_page: {snapshot.unknown_blocks} unknown block(s) in export")

    # The title, URL and created-time are connector-supplied text going
    # into a Markdown heading/bullet. The connector writes the RAW title
    # into the payload (matching granola.py's own write-time contract) and
    # never sanitises the url/created_time at all, so all three are
    # rendered inert here, at read time — the same belt-and-braces
    # ``extractors.meeting.extract`` uses for its own title. A
    # bracket-bearing URL gets mangled by this (``[``/``]`` become parens),
    # which is the right trade: an inert-but-ugly link beats a working
    # wikilink smuggled in through a "URL".
    lines = [f"# {safe_display_name(snapshot.title)}", ""]
    meta = []
    if snapshot.url:
        meta.append(f"**Notion URL:** {safe_display_name(snapshot.url)}")
    if snapshot.created_time:
        meta.append(f"**Created:** {safe_display_name(snapshot.created_time)}")
    if meta:
        lines += ["  \n".join(meta), ""]
    lines += ["## Content", "", snapshot.markdown if has_content else "_(no content)_", ""]

    return ExtractionResult(
        status=status, extractor="notion_page", markdown="\n".join(lines), notes=notes,
    )
