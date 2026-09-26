"""Notion connector: page listing, meeting-vs-note routing, incremental
pull via the cursor + refresh throttle, error containment, and the
``ConnectorError`` -> exit 2 mapping in ``scripts/pull.py``.

Fixtures are hand-built dicts shaped like real Notion API responses rather
than recorded real data, so these run offline and deterministically.
``notion._api_request`` (the connector's HTTP seam) is monkeypatched by a
small fake that dispatches on path and records every call, so tests can
assert exactly which requests did — and did not — happen.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.connectors import CONNECTORS, ConnectorError, load_state, run_connector, save_state
from ingest_lib.connectors import notion as notion_mod
from ingest_lib.connectors.state import ConnectorState
from ingest_lib.extractors import dispatch_extractor
from ingest_lib.extractors import meeting as meeting_ex
from ingest_lib.extractors import notion_page as notion_page_ex
from ingest_lib.notes import _split_frontmatter
from ingest_lib.pipeline import plan_ingest, run_ingest

# scripts/ has no __init__.py and isn't installed as a package (only
# scripts/ingest_lib is); pin it onto sys.path so pull.py is importable here.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pull  # noqa: E402

_LOG = logging.getLogger("test")
_AT = "2026-09-15T12:00:00Z"


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


# ------------------------------------------------------------- fixtures

def _rich_text(text: str) -> list[dict[str, object]]:
    return [{"type": "text", "plain_text": text, "text": {"content": text}}]


def _search_page(
    page_id: str, title: str, *, last_edited: str,
    created: str = "2026-01-01T00:00:00.000Z",
) -> dict[str, object]:
    return {
        "object": "page",
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "created_time": created,
        "last_edited_time": last_edited,
        "archived": False,
        "in_trash": False,
        "properties": {"Name": {"type": "title", "title": _rich_text(title)}},
    }


def _meeting_notes_block(
    *, status: str = "notes_ready", title: str = "Meeting",
    transcript_block_id: str | None = "transcript-1",
    attendees: list[object] | None = None,
    start_time: str = "2026-09-10T09:00:00.000Z",
) -> dict[str, object]:
    """A ``meeting_notes`` block shaped per live vendor docs (verified
    2026-09-15): ``calendar_event`` is a TOP-LEVEL field of the body, a
    sibling of ``children``, with ``start_time``/``end_time`` and
    ``attendees`` as user ids/names (see F1/F2 in the review batch)."""
    children: dict[str, object] = {}
    if transcript_block_id is not None:
        children["transcript_block_id"] = transcript_block_id
    return {
        "type": "meeting_notes",
        "has_children": False,
        "meeting_notes": {
            "title": _rich_text(title), "status": status, "children": children,
            "calendar_event": {
                "start_time": start_time, "end_time": "2026-09-10T09:30:00.000Z",
                "attendees": attendees if attendees is not None else [
                    {"name": "Alice Smith", "email": "alice@example.com"},
                    "Bob Jones",
                ],
            },
        },
    }


def _transcript_block(block_id: str, text: str, *, has_children: bool = False) -> dict[str, object]:
    return {
        "id": block_id, "type": "paragraph", "has_children": has_children,
        "paragraph": {"rich_text": _rich_text(text)},
    }


def _markdown_response(
    markdown: str, *, truncated: bool = False, unknown: list[str] | None = None
) -> dict[str, object]:
    return {"markdown": markdown, "truncated": truncated, "unknown_block_ids": unknown or []}


class FakeNotion:
    """Stands in for ``notion._api_request``: dispatches on path, records
    every call, and can be told to fail a specific block id or reject the
    token — all without touching the network."""

    def __init__(self) -> None:
        self.search_pages: list[dict[str, object]] = []
        self.blocks: dict[str, list[dict[str, object]]] = {}
        self.markdown: dict[str, dict[str, object]] = {}
        self.users: dict[str, dict[str, object]] = {}
        self.calls: list[str] = []
        self.fail_block_ids: set[str] = set()
        self.reject_token = False
        self.forbid_users = False

    def __call__(self, path: str, *, token: str, method: str = "GET", body: bytes | None = None) -> object:
        if path == "/search":
            self.calls.append("search")
            if self.reject_token:
                raise notion_mod._NotionRequestError("POST /search: HTTP 401", status=401)
            return {"results": self.search_pages, "has_more": False, "next_cursor": None}
        if path.startswith("/blocks/"):
            block_id = path[len("/blocks/"):].split("/", 1)[0]
            self.calls.append(f"blocks:{block_id}")
            if block_id in self.fail_block_ids:
                raise notion_mod._NotionRequestError(f"GET {path}: HTTP 500", status=500)
            return {"results": self.blocks.get(block_id, []), "has_more": False, "next_cursor": None}
        if path.startswith("/pages/"):
            rest = path[len("/pages/"):]
            page_id, _, query = rest.partition("/markdown")
            include_transcript = "true" if "include_transcript=true" in query else "false"
            key = f"{page_id}:{include_transcript}"
            self.calls.append(f"markdown:{key}")
            return self.markdown.get(key, {"markdown": "", "truncated": False, "unknown_block_ids": []})
        if path.startswith("/users/"):
            user_id = urllib.parse.unquote(path[len("/users/"):])
            self.calls.append(f"user:{user_id}")
            if self.forbid_users:
                raise notion_mod._NotionRequestError(f"GET {path}: HTTP 403", status=403)
            return self.users.get(user_id, {"name": None, "person": {}})
        raise AssertionError(f"unexpected path: {path}")


# --------------------------------------------------------------- item 1

def test_notion_registered_in_connectors_and_extractors() -> None:
    assert "notion" in CONNECTORS
    assert dispatch_extractor(Path("x.json"), relative_path="meetings/notion/x.json") is meeting_ex.extract
    assert dispatch_extractor(Path("x.json"), relative_path="notes/notion/x.json") is notion_page_ex.extract


# --------------------------------------------------------------- item 2

def test_missing_api_key_raises_connector_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    state = ConnectorState(name="notion")
    with pytest.raises(ConnectorError):
        list(notion_mod.NotionConnector().pull(state))


def test_pull_main_maps_connector_error_to_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    paths = _vault(tmp_path)
    monkeypatch.setattr(pull, "default_paths", lambda: paths)
    monkeypatch.setattr(pull, "_load_env", lambda: None)
    rc = pull.main(["notion"])
    assert rc == 2


def test_rejected_token_raises_connector_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "bad-token")
    fake = FakeNotion()
    fake.reject_token = True
    monkeypatch.setattr(notion_mod, "_api_request", fake)
    state = ConnectorState(name="notion")
    with pytest.raises(ConnectorError):
        list(notion_mod.NotionConnector().pull(state))


def test_invalid_refresh_days_raises_connector_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    monkeypatch.setenv("BRAIN_NOTION_REFRESH_DAYS", "not-a-number")
    state = ConnectorState(name="notion")
    with pytest.raises(ConnectorError):
        list(notion_mod.NotionConnector().pull(state))


# --------------------------------------------------------------- item 3

def test_note_page_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_id = "11111111-1111-1111-1111-111111111111"
    fake.search_pages = [_search_page(page_id, "Project Notes", last_edited="2026-09-10T10:00:00.000Z")]
    fake.blocks[page_id] = []  # no meeting_notes block -> note page
    fake.markdown[f"{page_id}:true"] = _markdown_response("# Project Notes\n\nSome content.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1

    sid = hashlib.sha1(page_id.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - test expectation, not security
    expected = f"inbox/notes/notion/project-notes-{sid}-20260910t100000.json"
    assert expected in stats.snapshots
    payload = json.loads((paths.root / expected).read_text())
    assert payload == {
        "connector": "notion", "id": page_id, "title": "Project Notes",
        "url": f"https://notion.so/{page_id}", "created_time": "2026-01-01T00:00:00.000Z",
        # last_edited_time is deliberately absent from the payload (F4) —
        # it lives only in the filename above.
        "markdown": "# Project Notes\n\nSome content.",
        "truncated": False, "unknown_blocks": 0,
    }

    content_calls_before = [c for c in fake.calls if c.startswith(("blocks:", "markdown:"))]
    stats2 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats2.written == 0
    content_calls_after = [c for c in fake.calls if c.startswith(("blocks:", "markdown:"))]
    # Not just "wrote nothing" — no content fetch (blocks or markdown) at
    # all happened on the second run.
    assert content_calls_after == content_calls_before


# --------------------------------------------------------------- item 4

def test_meeting_page_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_id = "22222222-2222-2222-2222-222222222222"
    fake.search_pages = [_search_page(page_id, "Weekly Sync", last_edited="2026-09-11T09:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(title="Weekly Sync AI Notes")]
    fake.blocks["transcript-1"] = [
        _transcript_block("t1", "Alice: let's start."),
        _transcript_block("t2", "Bob: sounds good."),
    ]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary text here.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    relpath = stats.snapshots[0]
    assert relpath.startswith("inbox/meetings/notion/")

    res = meeting_ex.extract(paths.root / relpath, tmp_path / "assets")
    assert res.status == "processed"
    assert "Alice: let's start." in res.markdown
    assert "Bob: sounds good." in res.markdown
    assert "[[knowledge/people/alice-smith]]" in res.markdown
    assert "[[knowledge/people/bob-jones]]" in res.markdown


# --------------------------------------------------------------- item 5

def test_meeting_with_unsettled_status_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3: an unsettled meeting is BLOCKED, not a freeze on the whole
    cursor — it holds back only pages behind it. With a second, ordinary
    page confirmed BEFORE it, the cursor advances to that page's
    ``last_edited_time`` instead of staying ``None`` forever."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    unsettled_id = "33333333-3333-3333-3333-333333333333"
    ok_id = "99999999-0000-0000-0000-000000000009"
    fake.search_pages = [
        _search_page(unsettled_id, "In Progress", last_edited="2026-09-11T09:00:00.000Z"),
        _search_page(ok_id, "Fine Page", last_edited="2026-09-05T09:00:00.000Z"),
    ]
    fake.blocks[unsettled_id] = [_meeting_notes_block(status="transcription_in_progress")]
    fake.blocks[ok_id] = []
    fake.markdown[f"{ok_id}:true"] = _markdown_response("ok content")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    assert load_state(paths, "notion").cursor == "2026-09-05T09:00:00.000Z"


# --------------------------------------------------------------- item 6

def test_cursor_advances_and_skips_already_fetched_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_a = "aaaaaaaa-0000-0000-0000-000000000001"
    page_b = "bbbbbbbb-0000-0000-0000-000000000002"
    fake.search_pages = [
        _search_page(page_a, "Page A", last_edited="2026-09-12T00:00:00.000Z"),
        _search_page(page_b, "Page B", last_edited="2026-09-10T00:00:00.000Z"),
    ]
    fake.blocks[page_a] = []
    fake.blocks[page_b] = []
    fake.markdown[f"{page_a}:true"] = _markdown_response("A content")
    fake.markdown[f"{page_b}:true"] = _markdown_response("B content")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats1 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats1.written == 2
    assert load_state(paths, "notion").cursor == "2026-09-12T00:00:00.000Z"

    fake.calls.clear()
    stats2 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats2.written == 0
    assert fake.calls == ["search"]  # unchanged listing -> no content fetches at all

    # A page newly shared with the integration (old last_edited_time, but
    # absent from state.entries) IS fetched despite predating the cursor.
    page_c = "cccccccc-0000-0000-0000-000000000003"
    fake.search_pages.append(_search_page(page_c, "Page C", last_edited="2020-01-01T00:00:00.000Z"))
    fake.blocks[page_c] = []
    fake.markdown[f"{page_c}:true"] = _markdown_response("C content")
    fake.calls.clear()
    stats3 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats3.written == 1
    assert f"blocks:{page_c}" in fake.calls
    assert f"blocks:{page_a}" not in fake.calls
    assert f"blocks:{page_b}" not in fake.calls
    # Cursor doesn't regress just because an old newly-shared page was pulled.
    assert load_state(paths, "notion").cursor == "2026-09-12T00:00:00.000Z"


# --------------------------------------------------------------- item 7

def test_refresh_throttle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # F3: build the fixture off the same monkeypatched clock the connector
    # itself reads (_now), not the live one — a fixed seam, not a race with
    # whatever time the test happens to run at.
    fixed_now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(notion_mod, "_now", lambda: fixed_now)

    paths = _vault(tmp_path)
    page_id = "44444444-0000-0000-0000-000000000004"
    two_days_ago = (fixed_now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = ConnectorState(name="notion", cursor="2026-09-01T00:00:00.000Z")
    state.entries[page_id] = {
        "content_hash": "deadbeef", "inbox_path": "inbox/notes/notion/old.json",
        "pulled_at": two_days_ago,
    }
    save_state(paths, state)

    fake = FakeNotion()
    fake.search_pages = [_search_page(page_id, "Throttled", last_edited="2026-09-14T00:00:00.000Z")]
    fake.blocks[page_id] = []
    fake.markdown[f"{page_id}:true"] = _markdown_response("New content")
    monkeypatch.setattr(notion_mod, "_api_request", fake)
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    monkeypatch.setenv("BRAIN_NOTION_REFRESH_DAYS", "7")

    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0
    assert f"blocks:{page_id}" not in fake.calls
    # The throttled page holds the cursor below its own last_edited_time
    # until its window lapses — it never advances the cursor to its lt.
    assert load_state(paths, "notion").cursor == "2026-09-01T00:00:00.000Z"

    monkeypatch.setenv("BRAIN_NOTION_REFRESH_DAYS", "0")
    fake.calls.clear()
    stats2 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats2.written == 1
    assert f"blocks:{page_id}" in fake.calls
    # Once the throttle window lapses the page is fetched, CONFIRMED, and
    # the cursor advances past it.
    assert load_state(paths, "notion").cursor == "2026-09-14T00:00:00.000Z"


# --------------------------------------------------------------- item 8

def test_per_page_failure_is_contained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    good_id = "55555555-0000-0000-0000-000000000005"
    bad_id = "66666666-0000-0000-0000-000000000006"
    fake.search_pages = [
        _search_page(good_id, "Good Page", last_edited="2026-09-13T00:00:00.000Z"),
        _search_page(bad_id, "Bad Page", last_edited="2026-09-14T00:00:00.000Z"),
    ]
    fake.blocks[good_id] = []
    fake.markdown[f"{good_id}:true"] = _markdown_response("Good content")
    fake.fail_block_ids.add(bad_id)
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    assert stats.snapshots[0].startswith("inbox/notes/notion/good-page-")
    # F3: the failed page BLOCKS only what is behind it — the cursor still
    # advances to the good page's (earlier) last_edited_time rather than
    # freezing at None.
    assert load_state(paths, "notion").cursor == "2026-09-13T00:00:00.000Z"


# -------------------------------------------------------------- item 10

def test_title_injection_is_neutralised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F12: the payload carries the RAW title (granola-parity — sanitising
    at write time would bypass ingest_lib.meetings' control-character
    refusal for the meeting path and lose archive fidelity); the note
    RENDERED from it must still be single-line and inert."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    malicious_title = "Mallory\n\n## Links\n\n- [[knowledge/people/ceo]]"
    fake = FakeNotion()
    page_id = "77777777-0000-0000-0000-000000000007"
    fake.search_pages = [_search_page(page_id, malicious_title, last_edited="2026-09-14T00:00:00.000Z")]
    fake.blocks[page_id] = []
    fake.markdown[f"{page_id}:true"] = _markdown_response("body")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    dest = paths.root / stats.snapshots[0]
    payload = json.loads(dest.read_text())
    assert payload["title"] == malicious_title  # raw, matching granola.py

    res = notion_page_ex.extract(dest, tmp_path / "assets")
    title_line = res.markdown.splitlines()[0]
    assert title_line.startswith("# ")
    assert "\n" not in title_line
    assert "[[" not in title_line


def test_meeting_title_injection_write_raw_render_sanitised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F12, meeting path: same write-raw/render-sanitised contract as the
    note-page test above, and as granola.py itself."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    malicious_title = "Mallory\n\n## Links\n\n- [[knowledge/people/ceo]]"
    fake = FakeNotion()
    page_id = "88888888-0000-0000-0000-000000000008"
    fake.search_pages = [_search_page(page_id, "Weekly Sync", last_edited="2026-09-14T00:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(title=malicious_title, transcript_block_id=None)]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary text.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    dest = paths.root / stats.snapshots[0]
    payload = json.loads(dest.read_text())
    assert payload["title"] == malicious_title  # raw, granola-parity

    res = meeting_ex.extract(dest, tmp_path / "assets")
    title_line = res.markdown.splitlines()[0]
    assert title_line.startswith("# ")
    assert "\n" not in title_line
    assert "[[" not in title_line


# ------------------------------------------------------------------- F2

def test_calendar_attendees_resolve_user_ids_and_literal_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per live vendor docs an attendee entry is a user id, not a name —
    resolved via GET /v1/users/{id}. Every prior shape still works: a dict
    already carrying a name/email, a dict carrying only an id, a bare
    user-id-shaped string, and a bare literal name."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_id = "cafe0000-0000-0000-0000-0000000000c1"
    user_id = "11112222-3333-4444-5555-666677778888"
    fake.users[user_id] = {"name": "Carol King", "person": {"email": "carol@example.com"}}
    fake.search_pages = [_search_page(page_id, "Standup", last_edited="2026-09-11T09:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(
        title="Standup",
        transcript_block_id=None,
        attendees=[
            {"name": "Alice Smith"},
            {"id": user_id},
            user_id,             # bare user-id-shaped string, also resolved
            "Bob Jones",         # bare literal name, used as-is
        ],
    )]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    payload = json.loads((paths.root / stats.snapshots[0]).read_text())
    assert payload["attendees"] == ["Alice Smith", "Carol King", "Carol King", "Bob Jones"]
    # Resolved once and cached — two lookups by the same id, one HTTP call.
    assert fake.calls.count(f"user:{user_id}") == 1


def test_attendee_email_fallback_when_user_has_no_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_id = "cafe0000-0000-0000-0000-0000000000c2"
    user_id = "22223333-4444-5555-6666-777788889999"
    fake.users[user_id] = {"name": None, "person": {"email": "dana@example.com"}}
    fake.search_pages = [_search_page(page_id, "Standup", last_edited="2026-09-11T09:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(
        title="Standup", transcript_block_id=None, attendees=[{"id": user_id}],
    )]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    payload = json.loads((paths.root / stats.snapshots[0]).read_text())
    assert payload["attendees"] == ["dana@example.com"]


def test_attendee_user_lookup_403_disables_resolver_without_failing_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolver failure is enrichment lost, never a page error: the
    snapshot still writes (sans that attendee) and the cursor still
    advances past this page."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    fake.forbid_users = True
    page_id = "dddd0000-0000-0000-0000-0000000000d1"
    user_id = "99998888-7777-6666-5555-444433332222"
    fake.search_pages = [_search_page(page_id, "Standup", last_edited="2026-09-11T09:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(
        title="Standup", transcript_block_id=None, attendees=[{"id": user_id}],
    )]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    payload = json.loads((paths.root / stats.snapshots[0]).read_text())
    assert payload["attendees"] == []  # unresolvable attendee is omitted, not invented
    assert load_state(paths, "notion").cursor == "2026-09-11T09:00:00.000Z"


def test_user_resolver_disables_after_first_403_and_stops_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNotion()
    fake.forbid_users = True
    monkeypatch.setattr(notion_mod, "_api_request", fake)
    resolver = notion_mod._UserResolver("t")

    assert resolver.display_name("11111111-1111-1111-1111-111111111111") is None
    assert resolver.disabled is True
    assert len(fake.calls) == 1
    # Disabled: a second, different id makes no further HTTP call.
    assert resolver.display_name("22222222-2222-2222-2222-222222222222") is None
    assert len(fake.calls) == 1


# ------------------------------------------------------------------- R3

def test_calendar_attendees_records_unresolved_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3: an attendee whose user-id lookup fails is not silently dropped
    from the payload with no trace — the payload records how many
    attendees it could not resolve (AGENTS.md rule 6) alongside the names
    it did resolve."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    fake.forbid_users = True
    page_id = "eeee0000-0000-0000-0000-0000000000e1"
    blocked_user_id = "77776666-5555-4444-3333-222211110000"
    fake.search_pages = [_search_page(page_id, "Standup", last_edited="2026-09-11T09:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(
        title="Standup", transcript_block_id=None,
        attendees=[
            "Alice Smith",             # bare literal name, no lookup needed
            {"id": blocked_user_id},   # needs a lookup, gets HTTP 403
        ],
    )]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    payload = json.loads((paths.root / stats.snapshots[0]).read_text())
    assert payload["attendees"] == ["Alice Smith"]
    assert payload["attendees_unresolved"] == 1


def test_calendar_attendees_unresolved_is_zero_when_all_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key is always present, even when nothing was unresolved — the
    payload's bytes must not depend on whether a run happened to hit a
    lookup failure."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_id = "eeee0000-0000-0000-0000-0000000000e2"
    fake.search_pages = [_search_page(page_id, "Standup", last_edited="2026-09-11T09:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(
        title="Standup", transcript_block_id=None, attendees=["Alice Smith", "Bob Jones"],
    )]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    payload = json.loads((paths.root / stats.snapshots[0]).read_text())
    assert payload["attendees_unresolved"] == 0


# ------------------------------------------------------------------- F3

def test_stale_error_block_stops_holding_cursor_and_becomes_reselectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: an ERROR/unsettled block older than the 14-day zombie-hold
    horizon no longer holds the cursor back — and when it carries a PRIOR
    entry (it was snapshotted once before it started failing/unsettling),
    that entry is popped from state.entries so the page reads exactly like
    one newly shared with the integration (``prior_entry is None``) and is
    fetched again on the very next run, regardless of the cursor. This is
    what makes the module docstring's "still retried every run" literally
    true rather than a permanently lost edit."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fixed_now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(notion_mod, "_now", lambda: fixed_now)
    paths = _vault(tmp_path)

    stale_id = "aaaa0000-0000-0000-0000-00000000000a"
    recent_id = "bbbb0000-0000-0000-0000-00000000000b"
    stale_lt = "2026-08-01T00:00:00.000Z"  # > 14 days before fixed_now

    state = ConnectorState(name="notion", cursor="2026-07-01T00:00:00.000Z")
    state.entries[stale_id] = {
        "content_hash": "deadbeef", "inbox_path": "inbox/notes/notion/old.json",
        "pulled_at": "2026-07-15T00:00:00.000Z",
    }
    save_state(paths, state)

    fake = FakeNotion()
    fake.search_pages = [
        _search_page(stale_id, "Stale Unsettled", last_edited=stale_lt),
        _search_page(recent_id, "Recent Page", last_edited="2026-09-14T00:00:00.000Z"),
    ]
    fake.blocks[stale_id] = [_meeting_notes_block(status="transcription_in_progress")]
    fake.blocks[recent_id] = []
    fake.markdown[f"{recent_id}:true"] = _markdown_response("recent content")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    state_after = load_state(paths, "notion")
    assert state_after.cursor == "2026-09-14T00:00:00.000Z"  # no longer held back
    assert stale_id not in state_after.entries  # its own entry was popped

    # Next run: with no prior entry, the stale page is fetched again
    # regardless of the cursor (prior_entry is None -> needs_fetch True) —
    # and a successful fetch re-records it.
    fake.calls.clear()
    fake.blocks[stale_id] = [_meeting_notes_block(status="notes_ready", transcript_block_id=None)]
    fake.markdown[f"{stale_id}:false"] = _markdown_response("Finally settled.")
    stats2 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert f"blocks:{stale_id}" in fake.calls
    assert stats2.written == 1
    assert stale_id in load_state(paths, "notion").entries  # re-recorded


def test_long_refresh_throttle_holds_cursor_past_14_day_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: a refresh-throttle block is NEVER horizon-dropped — with
    ``BRAIN_NOTION_REFRESH_DAYS=30``, a page throttled 20 days ago (already
    past the 14-day horizon that applies to ERROR blocks) still holds the
    cursor back for the rest of its window, and its state entry survives
    untouched. Cursor lag equal to the refresh window is the throttle
    working as designed, not data loss."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    monkeypatch.setenv("BRAIN_NOTION_REFRESH_DAYS", "30")
    fixed_now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(notion_mod, "_now", lambda: fixed_now)

    paths = _vault(tmp_path)
    page_id = "eeee0000-0000-0000-0000-00000000000e"
    twenty_days_ago = (fixed_now - timedelta(days=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = ConnectorState(name="notion", cursor="2026-08-01T00:00:00.000Z")
    state.entries[page_id] = {
        "content_hash": "deadbeef", "inbox_path": "inbox/notes/notion/old.json",
        "pulled_at": twenty_days_ago,
    }
    save_state(paths, state)

    fake = FakeNotion()
    fake.search_pages = [_search_page(page_id, "Long Throttle", last_edited="2026-09-13T00:00:00.000Z")]
    fake.blocks[page_id] = []
    fake.markdown[f"{page_id}:true"] = _markdown_response("edited content")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0
    assert f"blocks:{page_id}" not in fake.calls
    state_after = load_state(paths, "notion")
    assert state_after.cursor == "2026-08-01T00:00:00.000Z"  # held back, not advanced
    assert page_id in state_after.entries  # entry survives — throttle, not popped


def test_cursor_never_regresses_when_new_blocked_page_predates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_a = "cccc0000-0000-0000-0000-00000000000c"
    fake.search_pages = [_search_page(page_a, "Page A", last_edited="2026-09-12T00:00:00.000Z")]
    fake.blocks[page_a] = []
    fake.markdown[f"{page_a}:true"] = _markdown_response("A content")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats1 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats1.written == 1
    assert load_state(paths, "notion").cursor == "2026-09-12T00:00:00.000Z"

    # A newly-shared page, older than the cursor, whose meeting notes are
    # still unsettled — a real gap the connector must retry, but the
    # already-confirmed cursor must not fall back because of it.
    page_b = "dddd0000-0000-0000-0000-00000000000d"
    fake.search_pages.append(
        _search_page(page_b, "Old Unsettled", last_edited="2026-09-05T00:00:00.000Z")
    )
    fake.blocks[page_b] = [_meeting_notes_block(status="transcription_in_progress")]
    stats2 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats2.written == 0
    assert load_state(paths, "notion").cursor == "2026-09-12T00:00:00.000Z"


def test_meeting_date_fallback_is_validated(tmp_path: Path) -> None:
    """F8: the created_time fallback is validated the same way as the
    calendar candidate — an unvalidated slice would otherwise render a
    path-traversing inbox filename."""
    assert notion_mod._meeting_date({}, "../../../etc/x") == ""
    assert notion_mod._meeting_date({}, "2026-07-04T00:00:00.000Z") == "2026-07-04"


# ------------------------------------------------------------------- F6

def test_edited_meeting_repull_reconciles_into_one_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F6: the meeting filename embeds the page's ``last_edited_time`` so
    an edited re-pull lands beside the first version instead of hitting
    the pipeline's raw-archive-clash refusal — and ``ingest_lib.meetings``
    (verified by reading it: candidates are deduped by
    ``(node_id, native_id)``, newest wins) reconciles the two same-id
    snapshots into exactly one meeting note reflecting the newer content."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    fake = FakeNotion()
    page_id = "aaaaaaaa-1111-1111-1111-111111111111"
    fake.search_pages = [_search_page(page_id, "Weekly Sync", last_edited="2026-09-10T09:00:00.000Z")]
    fake.blocks[page_id] = [
        _meeting_notes_block(title="Weekly Sync AI Notes", transcript_block_id=None)
    ]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary v1.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats1 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats1.written == 1
    v1_path = stats1.snapshots[0]

    plan1 = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG)
    ingest_stats1 = run_ingest(paths, plan1, dry_run=False, logger=_LOG)
    assert ingest_stats1.manual_review == 0

    notes_dir = paths.knowledge / "meetings"
    notes = sorted(notes_dir.rglob("*.md"))
    assert len(notes) == 1
    assert "Summary v1." in notes[0].read_text(encoding="utf-8")

    # An edited re-pull: same page id, newer last_edited_time, new summary.
    monkeypatch.setenv("BRAIN_NOTION_REFRESH_DAYS", "0")
    fake.search_pages = [_search_page(page_id, "Weekly Sync", last_edited="2026-09-12T09:00:00.000Z")]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary v2, edited.")
    stats2 = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats2.written == 1
    v2_path = stats2.snapshots[0]
    assert v2_path != v1_path  # distinct archive path — F6's stamp, not a clash

    plan2 = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=_LOG)
    ingest_stats2 = run_ingest(paths, plan2, dry_run=False, logger=_LOG)
    assert ingest_stats2.manual_review == 0  # no false archive-clash

    notes = sorted(notes_dir.rglob("*.md"))
    assert len(notes) == 1  # reconciled to ONE note, not two — no conflict
    frontmatter, _body = _split_frontmatter(notes[0].read_text(encoding="utf-8"))
    # The note's provenance now names the v2 snapshot specifically — this
    # is what "reflecting v2" means here: ingest_lib.meetings refreshes the
    # derived source_id/source_file (and the "## Links" source bullet) on
    # a re-pull, but treats "## Notes" as hand-editable, human-owned
    # content it never overwrites (same as the granola precedent,
    # test_meetings.test_a_repulled_snapshot_updates_the_note_instead_of_conflicting).
    assert frontmatter["source_id"] == page_id
    assert frontmatter["source_file"] == f"archive/raw/{v2_path.removeprefix('inbox/')}"


# ------------------------------------------------------------------- R2

def test_long_meeting_title_yields_a_short_filename_and_pull_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: a 400-char Notion title must not reach ``meeting_filename``'s
    uncapped slugging — that would raise ``OSError: File name too long``
    and abort the whole pull. ``_versioned_meeting_filename`` pre-slugs the
    title via ``_note_slug`` (capped at 60 chars) first."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_abc")
    long_title = "Quarterly Planning Session " * 15  # ~420 chars
    assert len(long_title) > 400
    fake = FakeNotion()
    page_id = "ffff0000-0000-0000-0000-00000000000f"
    fake.search_pages = [_search_page(page_id, "Weekly Sync", last_edited="2026-09-14T00:00:00.000Z")]
    fake.blocks[page_id] = [_meeting_notes_block(title=long_title, transcript_block_id=None)]
    fake.markdown[f"{page_id}:false"] = _markdown_response("Summary.")
    monkeypatch.setattr(notion_mod, "_api_request", fake)

    paths = _vault(tmp_path)
    stats = run_connector(notion_mod.NotionConnector(), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    relpath = stats.snapshots[0]
    assert relpath.startswith("inbox/meetings/notion/")
    leaf = Path(relpath).name
    assert len(leaf) < 150
    assert (paths.root / relpath).exists()


# ------------------------------------------------------------------- F7

class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", code, "err", headers, None)


def test_api_request_retries_429_then_succeeds_honouring_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    attempts = {"n": 0}

    def fake_urlopen(req: object, timeout: float = 30) -> _FakeHTTPResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(429, retry_after="3")
        return _FakeHTTPResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = notion_mod._api_request("/x", token="t")
    assert result == {"ok": True}
    assert attempts["n"] == 2
    assert 3.0 in sleeps


def test_negative_retry_after_is_floored_and_never_sleeps_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    def record_sleep(s: float) -> None:
        assert s >= 0
        sleeps.append(s)

    monkeypatch.setattr(time, "sleep", record_sleep)
    attempts = {"n": 0}

    def fake_urlopen(req: object, timeout: float = 30) -> _FakeHTTPResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(429, retry_after="-5")
        return _FakeHTTPResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = notion_mod._api_request("/x", token="t")
    assert result == {"ok": True}
    assert 0.0 in sleeps


def test_api_request_529_retried_like_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def fake_urlopen(req: object, timeout: float = 30) -> _FakeHTTPResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(529)
        return _FakeHTTPResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = notion_mod._api_request("/x", token="t")
    assert result == {"ok": True}
    assert attempts["n"] == 2


def test_api_request_exhausts_retries_and_raises_with_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def fake_urlopen(req: object, timeout: float = 30) -> _FakeHTTPResponse:
        raise _http_error(429, retry_after="0")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(notion_mod._NotionRequestError) as exc_info:
        notion_mod._api_request("/x", token="t")
    assert exc_info.value.status == 429


def test_api_request_404_raises_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def fake_urlopen(req: object, timeout: float = 30) -> _FakeHTTPResponse:
        attempts["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(notion_mod._NotionRequestError) as exc_info:
        notion_mod._api_request("/x", token="t")
    assert exc_info.value.status == 404
    assert attempts["n"] == 1


def test_api_request_urlerror_retried_with_backoff_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def fake_urlopen(req: object, timeout: float = 30) -> _FakeHTTPResponse:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(notion_mod._NotionRequestError):
        notion_mod._api_request("/x", token="t")
    assert 1.0 in sleeps and 2.0 in sleeps and 4.0 in sleeps


def test_list_block_children_follows_pagination_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = {
        None: {"results": [{"id": "b1"}], "has_more": True, "next_cursor": "c2"},
        "c2": {"results": [{"id": "b2"}], "has_more": True, "next_cursor": "c3"},
        "c3": {"results": [{"id": "b3"}], "has_more": False, "next_cursor": None},
    }

    def fake_request(
        path: str, *, token: str, method: str = "GET", body: bytes | None = None
    ) -> object:
        cursor = path.split("start_cursor=", 1)[1] if "start_cursor=" in path else None
        return pages[cursor]

    monkeypatch.setattr(notion_mod, "_api_request", fake_request)
    result = notion_mod._list_block_children("blk", "t")
    assert [b["id"] for b in result] == ["b1", "b2", "b3"]


def test_list_block_children_stops_at_pagination_cap_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_request(
        path: str, *, token: str, method: str = "GET", body: bytes | None = None
    ) -> object:
        return {"results": [{"id": "x"}], "has_more": True, "next_cursor": "next"}

    monkeypatch.setattr(notion_mod, "_api_request", fake_request)
    with caplog.at_level(logging.WARNING, logger=notion_mod._LOG.name):
        result = notion_mod._list_block_children("blk", "t")
    assert len(result) == notion_mod._MAX_PAGINATION_PAGES
    assert any("truncated" in r.getMessage() for r in caplog.records)


def test_list_all_pages_paginates_and_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {"results": [{"id": "p1"}], "has_more": True, "next_cursor": "c2"},
        {"results": [{"id": "p2"}], "has_more": False, "next_cursor": None},
    ]
    calls: list[dict[str, object]] = []

    def fake_request(
        path: str, *, token: str, method: str = "GET", body: bytes | None = None
    ) -> object:
        assert path == "/search"
        payload = json.loads(body.decode("utf-8")) if body else {}
        calls.append(payload)
        return responses[len(calls) - 1]

    monkeypatch.setattr(notion_mod, "_api_request", fake_request)
    result = list(notion_mod._list_all_pages("t"))
    assert [p["id"] for p in result] == ["p1", "p2"]
    assert len(calls) == 2
    assert calls[1]["start_cursor"] == "c2"


def test_list_all_pages_stops_at_pagination_cap_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_request(
        path: str, *, token: str, method: str = "GET", body: bytes | None = None
    ) -> object:
        return {"results": [{"id": "p"}], "has_more": True, "next_cursor": "next"}

    monkeypatch.setattr(notion_mod, "_api_request", fake_request)
    with caplog.at_level(logging.WARNING, logger=notion_mod._LOG.name):
        result = list(notion_mod._list_all_pages("t"))
    assert len(result) == notion_mod._MAX_PAGINATION_PAGES
    assert any("truncated" in r.getMessage() for r in caplog.records)
