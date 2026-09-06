"""claude.ai / ChatGPT conversation-export connector (local-first).

Reads a user-downloaded ``conversations.json`` archive (claude.ai's "Export
data" or ChatGPT's "Export data") pointed at by ``BRAIN_CHAT_EXPORT_PATH``
(or the generic ``BRAIN_PULL_PATH`` that ``scripts/pull.py --path`` sets)
and normalises each conversation into the same transcript snapshot schema
as the Claude Code connector (see ``_transcript_common.py`` and
``extractors/transcript.py``), written to
``inbox/transcripts/chat_export/``. No API, no auth, no credential search —
this connector never looks for one, only reads the file it is pointed at.

**Snapshot unit: one conversation**, matching the Claude Code connector's
one-session-one-snapshot choice — the unit a human would search for later.
Native id is the export's own conversation id (stable across re-exports).

**Format is auto-detected from shape**, not from a filename convention
(both services name the file ``conversations.json``): a list of objects
carrying ``chat_messages`` is a claude.ai export; a list of objects
carrying ``mapping`` is a ChatGPT export (its DAG-of-edits tree — walked
here as "every node's message, ordered by ``create_time``", not just the
``current_node`` path, so an edited-and-regenerated branch isn't silently
dropped). An unrecognised shape is skipped rather than guessed at.

**What is dropped**, counted in ``stats`` per conversation: any message
whose role is neither ``user``/``human`` nor ``assistant`` (e.g. ChatGPT's
``system``/``tool`` authors), and any content part that is not plain text
(code-interpreter blocks, image parts, ChatGPT's non-``text`` content
types). Redaction and home-directory scrubbing are shared with the Claude
Code connector — see ``_transcript_common.py``.

**Volume.** A user's export can hold years of conversations. Bounded to the
``BRAIN_CHAT_EXPORT_MAX`` most recent conversations by default (newest
first) so importing an old, large archive for the first time cannot flood
``inbox/`` in one run; re-running with the same file is still idempotent
per-conversation via the connector's state (unchanged conversations are
skipped before they touch ``inbox/``). Each conversation's own total size is
separately bounded by ``BRAIN_TRANSCRIPT_MAX_CHARS`` (see
``_transcript_common.cap_session_turns``) so one huge conversation cannot
produce a megabyte-plus snapshot on its own.

**Source resolution.** ``BRAIN_PULL_PATH`` (set by ``pull.py --path``) wins
over ``BRAIN_CHAT_EXPORT_PATH`` when both are set, and a configured-but-
missing path raises rather than silently yielding nothing (see review M2).
A long conversation title is bounded before it becomes part of the inbox
filename (``filename_safe_title``) — the full title is still kept in the
snapshot/note (see review M1).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from ._transcript_common import (
    TranscriptSourceError,
    TranscriptStats,
    TranscriptTurn,
    build_transcript_payload,
    cap_session_turns,
    filename_safe_title,
    finalize_turn_text,
)
from .base import Snapshot, meeting_filename

if TYPE_CHECKING:
    from .state import ConnectorState

_PATH_ENV = "BRAIN_CHAT_EXPORT_PATH"
_GENERIC_PATH_ENV = "BRAIN_PULL_PATH"
_MAX_ENV = "BRAIN_CHAT_EXPORT_MAX"
_DEFAULT_MAX = 50


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _export_path() -> Path | None:
    """Resolve the export file to read.

    An explicit ``--path`` (which ``pull.py`` sets as ``BRAIN_PULL_PATH``)
    is the more specific, more recently expressed configuration, so it wins
    over the connector's own ``BRAIN_CHAT_EXPORT_PATH`` default rather than
    losing to it (review M2). No configuration at all yields ``None``
    (nothing to do); a path that IS configured but doesn't resolve to a file
    is a loud ``TranscriptSourceError`` rather than a silent zero-item pull
    indistinguishable from "nothing new".
    """
    generic = os.environ.get(_GENERIC_PATH_ENV)
    raw = generic or os.environ.get(_PATH_ENV)
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_file():
        env_name = _GENERIC_PATH_ENV if generic else _PATH_ENV
        raise TranscriptSourceError(
            f"{env_name} points at a path that is not a file: {p}"
        )
    return p


def _load_conversations(path: Path) -> list[object]:
    try:
        data: object = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _str_field(obj: dict[object, object], key: str) -> str:
    val = obj.get(key)
    return val if isinstance(val, str) else ""


def _claude_ai_message_text(m: dict[object, object], stats: TranscriptStats) -> str:
    text_field = m.get("text")
    if isinstance(text_field, str) and text_field.strip():
        return text_field
    content = m.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
        else:
            stats.dropped_non_text_parts += 1
    return "\n\n".join(parts)


def _emit_turn(
    turns: list[TranscriptTurn], stats: TranscriptStats,
    role: Literal["user", "assistant"], raw_text: str, home: str,
    *, already_counted_empty: bool = False,
) -> None:
    if not raw_text.strip():
        # Only count this as a dropped "wrapper turn" when nothing else
        # already explains the emptiness — a message whose only content
        # was non-text parts is already counted via dropped_non_text_parts
        # by the caller, and double-counting it here too made a rendered
        # note overstate how many turns were pure harness wrapper (review
        # N4, the chat_export analogue of the claude_code fix).
        if not already_counted_empty:
            stats.dropped_wrapper_turns += 1
        return
    text, n_redacted, truncated = finalize_turn_text(raw_text, home)
    if text is None:
        # Pure harness wrapper with no surviving prose (review C1/m3).
        stats.dropped_wrapper_turns += 1
        return
    stats.redacted_secrets += n_redacted
    if truncated:
        stats.truncated_long_turns += 1
    if role == "user":
        stats.kept_user_turns += 1
    else:
        stats.kept_assistant_turns += 1
    turns.append({"role": role, "text": text})


def _claude_ai_turns(obj: dict[object, object], home: str, stats: TranscriptStats) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    messages = obj.get("chat_messages")
    if not isinstance(messages, list):
        return turns
    for m in messages:
        if not isinstance(m, dict):
            continue
        sender = m.get("sender")
        if sender == "human":
            role: Literal["user", "assistant"] = "user"
        elif sender == "assistant":
            role = "assistant"
        else:
            stats.dropped_other_events += 1
            continue
        before = stats.dropped_non_text_parts
        raw_text = _claude_ai_message_text(m, stats)
        already_counted = stats.dropped_non_text_parts > before
        _emit_turn(turns, stats, role, raw_text, home, already_counted_empty=already_counted)
    return turns


def _chatgpt_turns(obj: dict[object, object], home: str, stats: TranscriptStats) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    mapping = obj.get("mapping")
    if not isinstance(mapping, dict):
        return turns
    entries: list[tuple[float, Literal["user", "assistant"], str]] = []
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author")
        role_raw = author.get("role") if isinstance(author, dict) else None
        if role_raw == "user":
            role: Literal["user", "assistant"] = "user"
        elif role_raw == "assistant":
            role = "assistant"
        else:
            stats.dropped_other_events += 1
            continue
        content = message.get("content")
        content_type = content.get("content_type") if isinstance(content, dict) else None
        if content_type != "text":
            stats.dropped_non_text_parts += 1
            continue
        parts_obj = content.get("parts") if isinstance(content, dict) else None
        parts: list[str] = []
        had_non_text_part = False
        if isinstance(parts_obj, list):
            for p in parts_obj:
                if isinstance(p, str):
                    parts.append(p)
                else:
                    stats.dropped_non_text_parts += 1
                    had_non_text_part = True
        raw_text = "\n\n".join(parts)
        if not raw_text.strip():
            # Only count as a "wrapper turn" when the emptiness isn't
            # already explained by dropped_non_text_parts above (review N4).
            if not had_non_text_part:
                stats.dropped_wrapper_turns += 1
            continue
        create_time = message.get("create_time")
        order_key = float(create_time) if isinstance(create_time, (int, float)) else 0.0
        entries.append((order_key, role, raw_text))
    entries.sort(key=lambda e: e[0])
    for _, role, raw_text in entries:
        _emit_turn(turns, stats, role, raw_text, home)
    return turns


def _native_id(obj: dict[object, object], turns: list[TranscriptTurn]) -> str:
    """Both known export formats always carry a stable id, but a malformed
    or future export might not — the fallback hashes the conversation's own
    turn content rather than its position in the export list, so re-running
    on an export with items reordered/prepended doesn't reshuffle every
    fallback id and break the connector's state idempotency key (review m5)."""
    for key in ("uuid", "conversation_id", "id"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    digest = hashlib.sha1(  # noqa: S324 - id tag, not security
        "\n".join(f"{t['role']}:{t['text']}" for t in turns).encode("utf-8")
    ).hexdigest()[:16]
    return f"conversation-{digest}"


def _date_from(obj: dict[object, object]) -> str:
    created = obj.get("created_at")
    if isinstance(created, str) and len(created) >= 10:
        return created[:10]
    create_time = obj.get("create_time")
    if isinstance(create_time, (int, float)):
        return datetime.fromtimestamp(float(create_time), tz=UTC).date().isoformat()
    return ""


class ChatExportConnector:
    name = "chat_export"

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]:
        path = _export_path()
        if path is None:
            return
        home = str(Path.home())
        max_items = _int_env(_MAX_ENV, _DEFAULT_MAX)

        dated: list[tuple[str, dict[object, object]]] = []
        for item in _load_conversations(path):
            if isinstance(item, dict):
                dated.append((_date_from(item), item))
        # Newest first, bounded — see module docstring's Volume note.
        dated.sort(key=lambda t: t[0], reverse=True)
        if max_items > 0:
            dated = dated[:max_items]

        for date, obj in dated:
            stats = TranscriptStats()
            connector_kind: str
            if "chat_messages" in obj:
                connector_kind = "claude_ai"
                turns = _claude_ai_turns(obj, home, stats)
                title = _str_field(obj, "name") or "Untitled conversation"
            elif "mapping" in obj:
                connector_kind = "chatgpt"
                turns = _chatgpt_turns(obj, home, stats)
                title = _str_field(obj, "title") or "Untitled conversation"
            else:
                continue
            if stats.kept_user_turns == 0 or stats.kept_assistant_turns == 0:
                continue
            native_id = _native_id(obj, turns)
            turns = cap_session_turns(turns, stats)
            if stats.kept_user_turns == 0 or stats.kept_assistant_turns == 0:
                # The head/tail cap (review N5) can still leave a
                # conversation with no real exchange in the surviving
                # window — re-run the guard already applied before capping.
                continue
            payload = build_transcript_payload(
                connector=connector_kind, native_id=native_id, title=title,
                date=date, context="", turns=turns, stats=stats,
            )
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            yield Snapshot(
                source_class="transcripts/chat_export", native_id=native_id,
                filename=meeting_filename(date, filename_safe_title(title), native_id),
                payload=body,
            )
