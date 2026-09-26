"""Notion connector.

Pulls every page a Notion internal integration can see, via the public API
(``api.notion.com``) with a hand-rolled ``urllib`` client — no new
dependency, the same choice ``granola.py`` makes. Two kinds of page become
two kinds of :class:`~.base.Snapshot`:

- A page whose top-level blocks include an AI meeting-notes block
  (``type == "meeting_notes"``) normalises into the SAME snapshot schema
  Granola/justREC write (see ``extractors/meeting.py``), plus one
  notion-only key, ``attendees_unresolved`` (a count of attendees the
  resolver could not name — see :func:`_calendar_attendees`); the shared
  extractor reads only the common keys and ignores the extra one. Filed
  under ``meetings/notion/`` so it rides the existing meeting extractor and
  promotion machinery unchanged.
- Every other page becomes a ``notes/notion/`` snapshot in a new schema,
  read by ``extractors/notion_page.py``.

Auth: ``NOTION_API_KEY`` from the environment. Missing/empty raises
:class:`~.base.ConnectorError` rather than yielding nothing — a
misconfigured connector must fail loudly, not look like "nothing new".

Pinned ``Notion-Version: 2026-03-11`` (see ``_NOTION_VERSION``) — the
meeting-notes block is named ``meeting_notes`` only under this API version
(verified against live vendor docs, 2026-09-15; see scripts/README.md).

**Incremental pull.** Notion's search has no server-side
``last_edited_time`` filter, so every run lists ALL pages the integration
can see (cheap) and decides per page whether to fetch its content, using
``state.cursor`` (the newest ``last_edited_time`` fully confirmed pulled)
plus ``state.entries`` (native id -> last snapshot). Listing everything
every run is deliberate, not wasteful: it is what catches a page newly
SHARED with the integration whose ``last_edited_time`` is old — Notion
access is granted per page in the UI and inherited by child pages, so a
page "new to us" can have been last edited long before we ever saw it.

**Refresh throttle.** A page already in ``state.entries`` whose last pull
is younger than ``BRAIN_NOTION_REFRESH_DAYS`` (default 7) is skipped this
run even if it has since been edited — otherwise a frequently-edited page
(a running doc, a personal dashboard) would land a brand-new immutable
archive version on every nightly pull. It stays eligible: the cursor is
only allowed to advance past a throttled page once it is actually
snapshotted (see :meth:`NotionConnector.pull`). A throttle block is NEVER
dropped for being old — it self-expires at a known time (``pulled_at`` plus
the refresh window), so by design it may hold the cursor back for up to the
whole window. Set ``BRAIN_NOTION_REFRESH_DAYS=30`` and a page throttled
20 days ago still holds the cursor: that page's cursor lag equals the
window, which is the throttle working as intended, not data loss — its own
``state.entries`` record is untouched throughout, so it is fetched right on
schedule the moment the window lapses.

**Cursor bookkeeping.** Every listed page is classified CONFIRMED (already
pulled and unedited since, or successfully snapshotted/yielded this run —
even one the runner later finds byte-identical and skips writing, which is
a runner-level dedupe, not evidence the page needs fetching again) or
BLOCKED; a gone/malformed page is skipped from both. BLOCKED splits into
two kinds that age differently: a refresh-throttle block (above — never
horizon-dropped) and an ERROR block, covering both a per-page fetch error
and an unsettled meeting (transcription not yet ``notes_ready``). The
cursor advances past every CONFIRMED page older than the oldest live block
— every throttle block plus every ERROR block still within
``_BLOCK_HOLD_MAX_DAYS`` (14 days) of "now" — so one permanently-stuck page
(an abandoned transcription that never reaches ``notes_ready``, a page that
fails every run) holds back only what is genuinely behind it, not the
whole sync forever. Past that horizon an ERROR block stops holding the
cursor back AND the connector pops that page's own ``state.entries`` record
(see :meth:`NotionConnector.pull`): with no prior entry left, the page reads
exactly like one newly shared with the integration
(``prior_entry is None``), so it is fetched again on every later run
regardless of where the cursor sits, until it either succeeds or settles —
this is what keeps a horizon-dropped page's edit from being lost for good,
rather than merely no longer blocking others while its own edit rots
unfetched. The cursor itself never regresses (see :meth:`NotionConnector.pull`).

**Per-version filenames.** A changed re-pull of a stable page under a
stable filename would collide with the pipeline's raw-archive-clash
refusal once the prior version has been archived and cleaned from
``inbox/`` (``pipeline.py``'s "refusing to overwrite" check) — so every
``notes/notion/`` snapshot's filename embeds a stamp derived from the
page's LISTING metadata (``last_edited_time`` as ``/search`` reports it),
not from the payload (see :func:`_note_filename`); every ``meetings/notion/``
snapshot's filename embeds the same stamp (see
:func:`_versioned_meeting_filename`), for the identical reason. The
``notes/notion/`` payload itself drops ``last_edited_time`` (see
:func:`_note_snapshot`): a timestamp-only edit (a property change, a page
move) would otherwise change the payload bytes — and so its content hash —
on every pull with no real content change, defeating the archive's
content-hash idempotency for a fresh immutable version and a paid LLM
summary over bytes the vault already has.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ..concepts import slugify
from .base import ConnectorError, Snapshot, meeting_filename, safe_display_name

if TYPE_CHECKING:
    from .state import ConnectorState

_LOG = logging.getLogger(__name__)

_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2026-03-11"
_TOKEN_ENV = "NOTION_API_KEY"
_REFRESH_DAYS_ENV = "BRAIN_NOTION_REFRESH_DAYS"
_DEFAULT_REFRESH_DAYS = 7

# An ERROR block (a per-page fetch error, or an unsettled meeting) only
# holds the cursor back while it is this recent — past it, a zombie page
# (see module docstring's "Cursor bookkeeping") stops degrading incremental
# sync for every other page, and the connector forgets it (pops its
# state.entries record) so it is fetched again next run. A refresh-throttle
# block is exempt from this cap entirely — see the module docstring's
# "Refresh throttle" note.
_BLOCK_HOLD_MAX_DAYS = 14

# Defensive cap on both the page-search pagination and any one block's
# children pagination — a runaway ``has_more`` chain (a misbehaving API, or
# a workspace with an absurd page count) must not loop forever.
_MAX_PAGINATION_PAGES = 200

# meeting-notes transcript blocks can nest (e.g. a bulleted line inside a
# toggle); three levels comfortably covers real transcript shapes without
# recursing indefinitely on a pathological block tree.
_TRANSCRIPT_MAX_DEPTH = 3

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_MAX_RETRY_AFTER_SECONDS = 60.0
# 180 req/min on standard Notion plans == 3 req/s; pace proactively rather
# than only reacting to a 429, so a big pull never even approaches the
# ceiling.
_REQUEST_PACE_SECONDS = 0.35

_NOTE_SLUG_MAX = 60
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIMESTAMP_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})")

# A Notion user id, as it appears bare in an older/alternate calendar-event
# attendee shape: 36-char dashed (8-4-4-4-12 hex) or 32-char bare hex.
_USER_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    r"|^[0-9a-fA-F]{32}$"
)


class _NotionRequestError(RuntimeError):
    """One Notion API request failed after retries. Caught per page (and
    once around the initial page listing) so a single bad request never
    aborts the whole pull. ``status`` lets a caller special-case a rejected
    token (401/403) from any other failure."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------- HTTP seam

def _retry_after_seconds(exc: urllib.error.HTTPError, attempt: int) -> float:
    header = exc.headers.get("Retry-After") if exc.headers is not None else None
    if header is not None:
        try:
            # Floored at 0 so a (malformed, negative) header can never make
            # time.sleep raise.
            return min(max(float(header), 0.0), _MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]


def _api_request(
    path: str, *, token: str, method: str = "GET", body: bytes | None = None
) -> object:
    """One Notion API call. Tests monkeypatch this — no network in CI.

    Retries up to ``_MAX_RETRIES`` times on 429/529, honouring
    ``Retry-After`` (capped at 60s) or falling back to 1s/2s/4s backoff.
    Any other HTTP status or transport error raises immediately after the
    retry budget — the caller decides what "cannot complete" means for its
    own context (bad token vs. a single page's fetch failing).
    """
    url = f"{_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }
    # Paced once per call, not once per retry attempt — a retry already
    # pays its own wait (Retry-After or the fixed backoff below).
    time.sleep(_REQUEST_PACE_SECONDS)
    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https API
                parsed: object = json.loads(resp.read().decode("utf-8"))
            return parsed
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 529) and attempt < _MAX_RETRIES:
                time.sleep(_retry_after_seconds(exc, attempt))
                continue
            raise _NotionRequestError(f"{method} {path}: HTTP {exc.code}", status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)])
                continue
            raise _NotionRequestError(f"{method} {path}: {exc}") from exc
    raise _NotionRequestError(f"{method} {path}: exhausted retries")


# ------------------------------------------------------- JSON narrowing

def _obj_dict(raw: object) -> dict[str, object]:
    """Narrow a decoded JSON value to a string-keyed dict, or ``{}`` if it
    isn't one. Every Notion response object is read this tolerantly."""
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items()}


def _obj_list(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    return [_obj_dict(item) for item in raw if isinstance(item, dict)]


def _str_field(obj: dict[str, object], key: str) -> str:
    val = obj.get(key)
    return val if isinstance(val, str) else ""


def _rich_text_plain(raw: object) -> str:
    parts: list[str] = []
    for item in _obj_list(raw):
        plain = item.get("plain_text")
        if isinstance(plain, str):
            parts.append(plain)
    return "".join(parts)


# ------------------------------------------------------------ page reads

def _page_id(page: dict[str, object]) -> str:
    return _str_field(page, "id")


def _is_gone(page: dict[str, object]) -> bool:
    return bool(page.get("archived")) or bool(page.get("in_trash"))


def _page_title(page: dict[str, object]) -> str:
    """The page's title property: the one whose value object has
    ``"type": "title"``, joined from its rich-text ``plain_text`` runs."""
    for value in _obj_dict(page.get("properties")).values():
        value_dict = _obj_dict(value)
        if value_dict.get("type") == "title":
            return _rich_text_plain(value_dict.get("title"))
    return ""


# ---------------------------------------------------------- HTTP calls

def _list_all_pages(token: str) -> Iterator[dict[str, object]]:
    """Every page object the integration can see, newest-edited first.

    No server-side ``last_edited_time`` filter exists on ``/v1/search``, so
    this always lists everything (see module docstring's "Incremental
    pull" note) — selection of what to actually FETCH happens in
    :meth:`NotionConnector.pull`, not here.
    """
    cursor: str | None = None
    for _ in range(_MAX_PAGINATION_PAGES):
        body: dict[str, object] = {
            "filter": {"property": "object", "value": "page"},
            "sort": {"timestamp": "last_edited_time", "direction": "descending"},
            "page_size": 100,
        }
        if cursor is not None:
            body["start_cursor"] = cursor
        raw = _api_request(
            "/search", token=token, method="POST", body=json.dumps(body).encode("utf-8")
        )
        data = _obj_dict(raw)
        yield from _obj_list(data.get("results"))
        if not data.get("has_more"):
            return
        next_cursor = data.get("next_cursor")
        if not isinstance(next_cursor, str):
            return
        cursor = next_cursor
    # Loop exhausted without has_more ever going false: some pages exist
    # beyond the cap and were silently never listed. Surfacing this beats
    # a truncated pull nobody heard about.
    _LOG.warning(
        "notion: page search truncated at the %d-page pagination cap "
        "(has_more was still true) — some pages were not listed this run",
        _MAX_PAGINATION_PAGES,
    )


def _list_block_children(block_id: str, token: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cursor: str | None = None
    for _ in range(_MAX_PAGINATION_PAGES):
        path = f"/blocks/{block_id}/children?page_size=100"
        if cursor is not None:
            path += f"&start_cursor={urllib.parse.quote(cursor, safe='')}"
        raw = _api_request(path, token=token)
        data = _obj_dict(raw)
        out.extend(_obj_list(data.get("results")))
        if not data.get("has_more"):
            return out
        next_cursor = data.get("next_cursor")
        if not isinstance(next_cursor, str):
            return out
        cursor = next_cursor
    # Same truncation signal as _list_all_pages above, for the other
    # pagination loop.
    _LOG.warning(
        "notion: block children listing for %s truncated at the %d-page "
        "pagination cap (has_more was still true) — some children were "
        "not listed this run",
        block_id, _MAX_PAGINATION_PAGES,
    )
    return out


def _page_markdown(page_id: str, token: str, *, include_transcript: bool) -> dict[str, object]:
    flag = "true" if include_transcript else "false"
    raw = _api_request(f"/pages/{page_id}/markdown?include_transcript={flag}", token=token)
    return _obj_dict(raw)


# --------------------------------------------------------- meeting-notes

def _find_meeting_notes_block(blocks: list[dict[str, object]]) -> dict[str, object] | None:
    """The page's TOP-LEVEL ``meeting_notes`` block, if any.

    Only top-level blocks are scanned — one nested inside a toggle, column,
    or other container is not found here, so that page routes to
    ``notes/notion/`` instead of meeting promotion. Its content and
    transcript are still captured there (the Markdown export inlines the
    transcript too, via ``include_transcript=true``); it just never
    reaches ``knowledge/meetings/`` (see scripts/README.md).
    """
    for block in blocks:
        if block.get("type") == "meeting_notes":
            return block
    return None


def _calendar_event_start(calendar_event: dict[str, object]) -> str:
    """The calendar event's start time. Per live vendor docs the field is
    ``start_time`` (a bare ISO 8601 string), checked first — but this shape
    has churned before, so the prior tolerant fallbacks stay: a bare
    ``start`` string, or an object under it carrying ``date_time``/``date``.
    """
    start_time = calendar_event.get("start_time")
    if isinstance(start_time, str):
        return start_time
    start = calendar_event.get("start")
    if isinstance(start, str):
        return start
    start_dict = _obj_dict(start)
    for key in ("date_time", "date"):
        val = start_dict.get(key)
        if isinstance(val, str):
            return val
    return ""


def _meeting_date(calendar_event: dict[str, object], created_time: str) -> str:
    candidate = _calendar_event_start(calendar_event)[:10]
    if _ISO_DATE_RE.match(candidate):
        return candidate
    fallback = created_time[:10]
    # Validated the same way as the calendar candidate above: an
    # unvalidated created_time slice (e.g. "../../../etc/x") would
    # otherwise reach meeting_filename() and render as a path-traversing
    # inbox filename.
    return fallback if _ISO_DATE_RE.match(fallback) else ""


class _UserResolver:
    """Resolves a Notion user id to a display name via ``GET
    /v1/users/{id}`` (top-level ``name``, falling back to
    ``person.email``), cached per id for the life of one :meth:`pull`.

    Requires the integration's "user information" capability; without it
    every lookup returns HTTP 403. The FIRST 403 disables all further
    lookups for the rest of this pull rather than re-attempting (and
    re-logging) on every remaining attendee of every remaining meeting.
    Resolver failures are never treated as page errors: attendee names are
    enrichment, not core content, so losing one must never hold back —
    let alone fail — the page's own snapshot, or the whole sync.
    """

    def __init__(self, token: str) -> None:
        self.token = token
        self.cache: dict[str, str | None] = {}
        self.disabled = False

    def display_name(self, user_id: str) -> str | None:
        if self.disabled:
            return None
        if user_id in self.cache:
            return self.cache[user_id]
        try:
            raw = _api_request(
                f"/users/{urllib.parse.quote(user_id, safe='')}", token=self.token
            )
        except _NotionRequestError as exc:
            if exc.status == 403:
                self.disabled = True
                _LOG.warning(
                    'notion: user lookup forbidden (HTTP 403) — grant the '
                    'integration the "user information" capability to '
                    "resolve attendee names; omitting attendee names for "
                    "the rest of this pull"
                )
            else:
                _LOG.warning("notion: could not resolve user %s (%r)", user_id, exc)
            self.cache[user_id] = None
            return None
        user = _obj_dict(raw)
        name = user.get("name")
        resolved: str | None
        if isinstance(name, str) and name.strip():
            resolved = name
        else:
            email = _obj_dict(user.get("person")).get("email")
            resolved = email if isinstance(email, str) and email.strip() else None
        self.cache[user_id] = resolved
        return resolved


def _calendar_attendees(
    calendar_event: dict[str, object], resolver: _UserResolver
) -> tuple[list[str], int]:
    """Attendee display names from ``calendar_event["attendees"]``, plus a
    count of entries that needed a resolver lookup and didn't get one.

    Per live vendor docs an entry is a USER ID, not a name — but this shape
    has churned before, so every prior form is still read tolerantly: a
    dict already carrying ``name``/``email`` is used directly; a dict
    carrying only an ``id``, or a bare string shaped like a Notion user id
    (36-char dashed hex or 32-char bare hex), is resolved via ``resolver``;
    any other bare string is treated as a literal display name. Every
    resolved name is rendered inert via :func:`safe_display_name`.

    An attendee whose lookup fails (403-disabled resolver, or any other
    lookup failure — see :meth:`_UserResolver.display_name`) is omitted
    from the returned names, never invented — but it is still counted in
    the returned ``unresolved`` total, so the caller can record honestly
    that the attendee list is incomplete (AGENTS.md rule 6) rather than
    silently shrinking it.
    """
    raw = calendar_event.get("attendees")
    if not isinstance(raw, list):
        return [], 0
    out: list[str] = []
    unresolved = 0
    for entry in raw:
        name: str | None = None
        needs_lookup = False
        if isinstance(entry, dict):
            entry_dict = _obj_dict(entry)
            n = entry_dict.get("name")
            e = entry_dict.get("email")
            if isinstance(n, str) and n.strip():
                name = n
            elif isinstance(e, str) and e.strip():
                name = e
            else:
                uid = entry_dict.get("id")
                if isinstance(uid, str) and uid:
                    needs_lookup = True
                    name = resolver.display_name(uid)
        elif isinstance(entry, str):
            if _USER_ID_RE.match(entry):
                needs_lookup = True
                name = resolver.display_name(entry)
            else:
                name = entry
        if needs_lookup and name is None:
            unresolved += 1
            continue
        if name is not None:
            clean = safe_display_name(name)
            if clean:
                out.append(clean)
    return out, unresolved


def _block_rich_text(block: dict[str, object]) -> str | None:
    btype = block.get("type")
    if not isinstance(btype, str):
        return None
    rich_text = _obj_dict(block.get(btype)).get("rich_text")
    if not isinstance(rich_text, list):
        return None
    return _rich_text_plain(rich_text)


def _fetch_transcript_lines(block_id: str, token: str, *, depth: int) -> list[str]:
    """One line per block under ``block_id``, in document order, recursing
    into ``has_children`` up to ``depth`` levels. A block without a
    ``rich_text`` list contributes no line of its own but is still
    recursed into."""
    lines: list[str] = []
    for block in _list_block_children(block_id, token):
        text = _block_rich_text(block)
        if text is not None:
            lines.append(text)
        if depth > 0 and block.get("has_children"):
            child_id = _str_field(block, "id")
            if child_id:
                lines.extend(_fetch_transcript_lines(child_id, token, depth=depth - 1))
    return lines


def _meeting_snapshot(
    page: dict[str, object], block: dict[str, object], token: str, resolver: _UserResolver
) -> Snapshot | None:
    """Build the meeting-schema snapshot for a page carrying a
    ``meeting_notes`` block, or ``None`` when its transcription isn't
    settled yet (half-baked; the page re-selects on a later run)."""
    page_id = _page_id(page)
    body = _obj_dict(block.get("meeting_notes"))
    status = _str_field(body, "status")
    if status != "notes_ready":
        _LOG.warning(
            "notion: page %s meeting notes not ready (status=%r) — skipping this run",
            page_id, status,
        )
        return None

    # ``calendar_event`` is a TOP-LEVEL field of the ``meeting_notes`` body
    # — a sibling of ``children``, not nested inside it — per live vendor
    # docs (verified 2026-09-15). This shape has churned before, so treat
    # any future move as expected rather than surprising.
    calendar_event = _obj_dict(body.get("calendar_event"))
    children_obj = _obj_dict(body.get("children"))
    created_time = _str_field(page, "created_time")
    date = _meeting_date(calendar_event, created_time)

    block_title = _rich_text_plain(body.get("title"))
    # Raw, matching granola.py's own write-time contract (F12 in the review
    # batch): the title is sanitised at RENDER time
    # (extractors.meeting.extract), never here — pre-sanitising it here
    # would bypass ingest_lib.meetings' control-character refusal and lose
    # archive fidelity.
    title = block_title or _page_title(page) or "Untitled meeting"
    attendees, attendees_unresolved = _calendar_attendees(calendar_event, resolver)

    transcript_block_id = _str_field(children_obj, "transcript_block_id")
    if transcript_block_id:
        transcript = "\n".join(
            _fetch_transcript_lines(transcript_block_id, token, depth=_TRANSCRIPT_MAX_DEPTH)
        )
    else:
        transcript = ""
    summary = _str_field(_page_markdown(page_id, token, include_transcript=False), "markdown")

    normalized = {
        "connector": "notion",
        "id": page_id,
        "title": title,
        "date": date,
        "attendees": attendees,
        # Notion-only key (not part of the Granola/justREC shared schema —
        # extractors/meeting.py ignores unknown keys): how many attendees
        # a lookup failure omitted from `attendees` above, so the payload
        # never claims a complete list it doesn't have (AGENTS.md rule 6).
        # Always present, 0 when none, so the payload's bytes never depend
        # on whether this run happened to hit a lookup failure.
        "attendees_unresolved": attendees_unresolved,
        "summary": summary.strip(),
        "transcript": transcript.strip(),
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Snapshot(
        source_class="meetings/notion", native_id=page_id,
        filename=_versioned_meeting_filename(
            date, title, page_id, _str_field(page, "last_edited_time")
        ),
        payload=payload,
    )


# ------------------------------------------------------------- note pages

def _compact_stamp(iso: str) -> str:
    """``last_edited_time`` compacted to ``YYYYMMDDtHHMMSS``. Every content
    version of a note or meeting page gets a distinct filename (see module
    docstring's "Per-version filenames" note); a timestamp that fails to
    parse (never observed from the real API) falls back to an all-zero
    stamp rather than raising."""
    m = _TIMESTAMP_RE.match(iso)
    if not m:
        return "00000000t000000"
    y, mo, d, h, mi, s = m.groups()
    return f"{y}{mo}{d}t{h}{mi}{s}"


def _note_slug(title: str) -> str:
    slug = slugify(title) or "page"
    return slug[:_NOTE_SLUG_MAX].strip("-") or "page"


def _note_filename(title: str, page_id: str, last_edited_time: str) -> str:
    slug = _note_slug(title)
    sid = hashlib.sha1(page_id.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - filename tag, not security
    return f"{slug}-{sid}-{_compact_stamp(last_edited_time)}.json"


def _versioned_meeting_filename(date: str, title: str, native_id: str, last_edited_time: str) -> str:
    """``meeting_filename(...)`` with the page's ``last_edited_time`` stamp
    inserted before the extension. A stable meeting filename would collide
    with the pipeline's raw-archive-clash refusal once an edited re-pull's
    prior inbox copy has already been cleaned — the same problem
    :func:`_note_filename` solves for note pages, for the same reason.
    ``ingest_lib.meetings`` reconciles snapshots by native id regardless of
    which archive path holds them (candidates are deduped by
    ``(node_id, native_id)``, newest wins), so this only changes which path
    holds each content version, never how many meeting notes a re-pull
    produces.

    ``title`` is pre-slugged via :func:`_note_slug` (capped at
    ``_NOTE_SLUG_MAX`` chars) before reaching :func:`meeting_filename`,
    whose own slugging is uncapped — an unbounded Notion page title (up to
    2,000 chars) would otherwise produce a filename long enough to raise
    ``OSError: File name too long`` and abort the whole pull. Slugging is
    idempotent, so this changes nothing for a normal short title: an
    already-slugged string re-slugs to itself.
    """
    base = meeting_filename(date, _note_slug(title), native_id)
    stem = base[: -len(".json")]
    return f"{stem}-{_compact_stamp(last_edited_time)}.json"


def _note_snapshot(page: dict[str, object], token: str) -> Snapshot:
    page_id = _page_id(page)
    # Raw, matching granola.py's own write-time contract (F12): sanitised
    # at RENDER time in extractors/notion_page.py instead, not here.
    title = _page_title(page) or "Untitled page"
    last_edited_time = _str_field(page, "last_edited_time")
    markdown_response = _page_markdown(page_id, token, include_transcript=True)
    unknown_ids = markdown_response.get("unknown_block_ids")

    normalized = {
        "connector": "notion",
        "id": page_id,
        "title": title,
        "url": _str_field(page, "url"),
        "created_time": _str_field(page, "created_time"),
        # last_edited_time is deliberately NOT in the payload (F4): a
        # timestamp-only edit (a property change, a page move) must not
        # change the payload bytes with no real content change — the
        # FILENAME below still carries it, derived from the page's own
        # listing metadata rather than from the payload (see module
        # docstring's "Per-version filenames" note).
        "markdown": _str_field(markdown_response, "markdown"),
        "truncated": bool(markdown_response.get("truncated")),
        "unknown_blocks": len(unknown_ids) if isinstance(unknown_ids, list) else 0,
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Snapshot(
        source_class="notes/notion", native_id=page_id,
        filename=_note_filename(title, page_id, last_edited_time), payload=payload,
    )


def _snapshot_for_page(page: dict[str, object], token: str, resolver: _UserResolver) -> Snapshot | None:
    blocks = _list_block_children(_page_id(page), token)
    meeting_block = _find_meeting_notes_block(blocks)
    if meeting_block is not None:
        return _meeting_snapshot(page, meeting_block, token, resolver)
    return _note_snapshot(page, token)


# ------------------------------------------------------------------ env

def _refresh_days() -> int:
    raw = os.environ.get(_REFRESH_DAYS_ENV)
    if raw is None:
        return _DEFAULT_REFRESH_DAYS
    try:
        return int(raw)
    except ValueError as exc:
        raise ConnectorError(
            f"notion: {_REFRESH_DAYS_ENV} must be an integer, got {raw!r}"
        ) from exc


# --------------------------------------------------------- cursor math

def _now() -> datetime:
    """The clock seam for both the refresh window and the zombie-hold
    horizon below — one seam so a test can monkeypatch both at once."""
    return datetime.now(UTC)


def _iso_ms(dt: datetime) -> str:
    """``dt`` formatted like Notion's own timestamps (millisecond
    precision, ``Z`` suffix) so it compares lexicographically against a
    real ``last_edited_time`` the same safe way the cursor itself does."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _max_iso(a: str | None, b: str | None) -> str | None:
    """None-aware max of two ISO timestamps (plain string compare — see
    the lexicographic-safety comment in :meth:`NotionConnector.pull`);
    never regresses a real cursor value back to ``None``."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


def _within_refresh_window(pulled_at: str, refresh_days: int) -> bool:
    try:
        parsed = datetime.fromisoformat(pulled_at)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return _now() - parsed < timedelta(days=refresh_days)


class NotionConnector:
    name = "notion"

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]:
        token = os.environ.get(_TOKEN_ENV, "").strip()
        if not token:
            raise ConnectorError(
                f"notion: {_TOKEN_ENV} is not set — see scripts/README.md's notion connector section"
            )
        refresh_days = _refresh_days()

        try:
            pages = list(_list_all_pages(token))
        except _NotionRequestError as exc:
            if exc.status in (401, 403):
                raise ConnectorError(
                    f"notion: {_TOKEN_ENV} was rejected by the Notion API "
                    f"(HTTP {exc.status}) — see scripts/README.md"
                ) from exc
            raise ConnectorError(f"notion: could not list pages ({exc})") from exc

        old_cursor = state.cursor
        # One resolver per pull(): attendee-name lookups are cached for the
        # life of this run, and a 403 disables the resolver for the rest of
        # it (see _UserResolver) rather than per meeting.
        resolver = _UserResolver(token)
        # Every listed page is classified CONFIRMED or BLOCKED (see module
        # docstring's "Cursor bookkeeping"); gone/malformed pages land in
        # neither list. BLOCKED splits into throttle (self-expiring, never
        # horizon-dropped) and error (fetch failure or unsettled meeting;
        # tracked with its page_id so a horizon-dropped one can be popped
        # from state.entries below).
        confirmed_lts: list[str] = []
        blocked_throttle_lts: list[str] = []
        blocked_error: list[tuple[str, str]] = []

        for page in pages:
            if _is_gone(page):
                continue
            page_id = _page_id(page)
            last_edited = _str_field(page, "last_edited_time")
            if not page_id or not last_edited:
                continue

            prior_entry = state.entries.get(page_id)
            needs_fetch = prior_entry is None or old_cursor is None or last_edited > old_cursor
            if not needs_fetch:
                # Already fully pulled and not edited since — CONFIRMED
                # without a fetch.
                confirmed_lts.append(last_edited)
                continue

            if prior_entry is not None:
                pulled_at = prior_entry.get("pulled_at", "")
                if pulled_at and _within_refresh_window(pulled_at, refresh_days):
                    # Refresh throttle (see module docstring) — BLOCKED,
                    # self-expiring; re-eligible once the throttle window
                    # lapses, and NEVER horizon-dropped below (its own
                    # window is the bound, however long that is).
                    blocked_throttle_lts.append(last_edited)
                    continue

            try:
                snap = _snapshot_for_page(page, token, resolver)
            except _NotionRequestError as exc:
                _LOG.warning("notion: fetch failed for page %s (%r) — skipping this run", page_id, exc)
                blocked_error.append((page_id, last_edited))
                continue

            if snap is None:
                # Meeting page with unsettled transcription — already
                # logged in _meeting_snapshot. An ERROR block, not a
                # throttle: it is retried every run until it settles or
                # ages past the zombie-hold horizon below.
                blocked_error.append((page_id, last_edited))
                continue

            yield snap
            # CONFIRMED from the connector's own view even if the runner
            # later finds this snapshot byte-identical to one already
            # archived and skips writing it — that is a runner-level
            # dedupe, not evidence this page needs fetching again.
            confirmed_lts.append(last_edited)

        # An ERROR block only holds the cursor back while it is recent —
        # past _BLOCK_HOLD_MAX_DAYS a zombie page (an abandoned
        # transcription, a page that fails every run) stops degrading
        # incremental sync for every other page. Past the horizon we also
        # pop the page's own state.entries record (guarded: only if
        # present) so `prior_entry is None` on every later run — the page
        # then reads exactly like one newly shared with the integration and
        # is fetched again regardless of the cursor, which is what makes it
        # "still retried every run" rather than a permanently lost edit.
        # Mutating the connector's own entries here is sound: the runner
        # only ever reads them via `is_unchanged`/`record`, and the worst
        # case of popping one that then fetches unchanged is a single
        # redundant re-fetch whose identical snapshot re-creates the same
        # entry (same filename, same content hash — the pipeline's
        # equal-hash raw re-archive is a no-op).
        horizon = _iso_ms(_now() - timedelta(days=_BLOCK_HOLD_MAX_DAYS))
        recent_error_lts: list[str] = []
        for error_page_id, error_lt in blocked_error:
            if error_lt >= horizon:
                recent_error_lts.append(error_lt)
            else:
                state.entries.pop(error_page_id, None)
        live_blocks = blocked_throttle_lts + recent_error_lts
        min_block = min(live_blocks) if live_blocks else None
        computed: str | None = None
        for lt in confirmed_lts:
            if (min_block is None or lt < min_block) and (computed is None or lt > computed):
                computed = lt
        # Comparison throughout is a plain string compare of ISO-8601 UTC
        # timestamps, which is lexicographically safe (fixed-width fields,
        # common precision, common "Z" suffix); the cursor never regresses.
        state.cursor = _max_iso(old_cursor, computed)
