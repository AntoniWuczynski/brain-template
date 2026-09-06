"""Security-boundary regression tests for the MCP write/read surface.

These pin the fixes for the traversal, existence-oracle, profile-budget-
bypass, and query-DoS findings. They drive the real tools over a throwaway
git vault with background workers disabled — no live server, fully offline.
"""
from __future__ import annotations

import contextvars
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from ingest_lib.relations import is_valid_node_id

from mcp_server import tools as tools_mod
from mcp_server import tools_read as tools_read_mod
from mcp_server.audit import AuditLog
from mcp_server.safety import (
    SafetyError,
    resolve_inbox,
    resolve_read,
    resolve_write_under_allowlist,
)
from mcp_server.config import CONCEPT_WRITE_PREFIX, ServerConfig, WRITE_RATE_PER_MINUTE
from mcp_server.entity_tools import (
    tool_entity_append_fact,
    tool_entity_upsert_relation,
    tool_meeting_create,
)
from mcp_server.errors import ToolError
from mcp_server.identity import AGENT_VAR, current_agent
from mcp_server.memory_tools import tool_profile_update
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import tool_update_concept_user_section
from mcp_server.tools_read import (
    MAX_QUERY_CHARS,
    tool_list,
    tool_related,
    tool_search,
)


# ---------------------------------------------------------------- harness

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _make_vault(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    return root


def _cfg(root: Path) -> ServerConfig:
    return ServerConfig(
        vault_root=root,
        tokens=(("x" * 24, "agent-a"),),
        bind_host="127.0.0.1",
        bind_port=0,
        git_push_on_write=False,
        git_remote="origin",
        git_branch="main",
        log_level="warning",
        allowed_hosts=(),
        profile_max_bytes=4096,
    )


def _runtime(root: Path) -> Runtime:
    audit = AuditLog(root)
    return Runtime(
        audit=audit,
        push_worker=PushWorker(root, remote="origin", branch="main", enabled=False),
        refresher=IndexRefresher(root, audit=audit, enabled=False),
    )


@pytest.fixture()
def env(tmp_path: Path) -> Iterator[tuple[Path, ServerConfig, Runtime]]:
    root = _make_vault(tmp_path)
    cfg = _cfg(root)
    runtime = _runtime(root)
    token = AGENT_VAR.set("agent-a")
    try:
        yield root, cfg, runtime
    finally:
        AGENT_VAR.reset(token)


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ------------------------------------------------------- is_valid_node_id

@pytest.mark.parametrize("good", [
    "people/anna-kowalska",
    "organisations/acme",
    "meetings/2026/2026-06-12-kern-call",
    "projects/fyp",
])
def test_valid_node_ids_accepted(good: str) -> None:
    assert is_valid_node_id(good)


@pytest.mark.parametrize("bad", [
    "",                                   # empty
    "people",                             # no slash
    "people/../../archive/processed/x",   # traversal
    "people/./x",                         # dot segment
    "/people/x",                          # leading slash
    "../people/x",                        # leading ..
    "people/x/",                          # trailing slash -> empty segment
    "People/X",                           # uppercase (node ids are lowercase)
    "people/a b",                         # space
])
def test_invalid_node_ids_rejected(bad: str) -> None:
    assert not is_valid_node_id(bad)


# ---------------------------------- F000: meeting_create traversal blocked

def test_meeting_create_rejects_traversal_attendee(env: tuple[Path, ServerConfig, Runtime]) -> None:
    root, cfg, runtime = env
    # A ground-truth file the attacker would try to rewrite.
    _write(root, "archive/processed/university/secret.md", "# secret\n\nground truth\n")
    before = (root / "archive/processed/university/secret.md").read_text(encoding="utf-8")
    with pytest.raises(ToolError, match="people/ node id"):
        tool_meeting_create(
            cfg, runtime,
            date="2026-07-03", title="x",
            attendees=["people/../../archive/processed/university/secret"],
        )
    # The archive file is untouched — the write allowlist held.
    assert (root / "archive/processed/university/secret.md").read_text(encoding="utf-8") == before


# ------------------------- F074: relation target traversal blocked

def test_upsert_relation_rejects_traversal_target(env: tuple[Path, ServerConfig, Runtime]) -> None:
    root, cfg, runtime = env
    _write(root, "knowledge/people/anna.md", "---\ntitle: Anna\n---\n\n## Log\n")
    with pytest.raises(ToolError, match="valid node id"):
        tool_entity_upsert_relation(
            cfg, runtime,
            entity_path="knowledge/people/anna.md",
            rel="works_at",
            target="people/../../archive/processed/university/secret",
        )


# ------------------------- F011: source path is not an existence oracle

def test_append_fact_source_oracle_collapsed(env: tuple[Path, ServerConfig, Runtime]) -> None:
    root, cfg, runtime = env
    _write(root, "knowledge/people/anna.md", "---\ntitle: Anna\n---\n\n## Log\n")
    _write(root, ".env", "SECRET=hunter2\n")  # exists, outside knowledge/ + archive/
    # A '..' escaping the declared prefix must read as 'does not exist',
    # identical to a genuinely absent path — no file-existence oracle.
    with pytest.raises(ToolError, match="source does not exist") as esc:
        tool_entity_append_fact(
            cfg, runtime,
            entity_path="knowledge/people/anna.md",
            text="leaked", source="archive/../.env",
        )
    with pytest.raises(ToolError, match="source does not exist") as absent:
        tool_entity_append_fact(
            cfg, runtime,
            entity_path="knowledge/people/anna.md",
            text="x", source="archive/nope.pdf",
        )
    # Both messages are the same shape -> the oracle is collapsed.
    assert "does not exist" in str(esc.value)
    assert "does not exist" in str(absent.value)


# ------------------- F012: entity verbs can't grow PROFILE.md past its budget

def test_entity_append_fact_refuses_profile(env: tuple[Path, ServerConfig, Runtime]) -> None:
    root, cfg, runtime = env
    _write(root, "knowledge/people/anna.md", "---\ntitle: Anna\n---\n\n## Log\n")
    tool_profile_update(cfg, runtime, content="# Profile\n\nLikes uv.\n")
    with pytest.raises(ToolError, match="profile_update"):
        tool_entity_append_fact(
            cfg, runtime,
            entity_path="knowledge/assistant/PROFILE.md",
            text="x" * 400, source="knowledge/people/anna",
        )


# ------------------- F083/F084: query length is bounded

def test_search_rejects_oversized_query(env: tuple[Path, ServerConfig, Runtime]) -> None:
    root, cfg, runtime = env
    with pytest.raises(ToolError, match="query is"):
        tool_search(cfg, runtime, query="x" * (MAX_QUERY_CHARS + 1))


def test_related_rejects_oversized_concept(env: tuple[Path, ServerConfig, Runtime]) -> None:
    root, cfg, runtime = env
    with pytest.raises(ToolError, match="query is"):
        tool_related(cfg, runtime, concept="x" * (MAX_QUERY_CHARS + 1))


# ---------------- F001/F015: safety primitives (pure, tmp_path, no server)

def _vault_root(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "v"
    (root / "knowledge" / "notes").mkdir(parents=True)
    (root / "archive" / "processed").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    return root


def test_resolve_read_allows_and_denies(tmp_path: Path) -> None:
    root = _vault_root(tmp_path)
    (root / "knowledge" / "notes" / "a.md").write_text("x", encoding="utf-8")
    # Allowed prefix + root doc file.
    assert resolve_read(root, "knowledge/notes/a.md").name == "a.md"
    (root / "README.md").write_text("x", encoding="utf-8")
    assert resolve_read(root, "README.md").name == "README.md"
    # Outside the read allowlist (server code, scripts/).
    with pytest.raises(SafetyError, match=r"not found or not readable"):
        resolve_read(root, "mcp_server/tools.py")
    # Deny-listed within an allowed area (logs, .env, the embeddings index).
    for denied in ("logs/x.log", ".env", "metadata/embeddings_meta.jsonl"):
        with pytest.raises(SafetyError, match=r"not found or not readable"):
            resolve_read(root, denied)


def test_resolve_read_deny_is_casefolded(tmp_path: Path) -> None:
    # On a case-insensitive FS the DENY check must casefold, or the
    # embeddings text leaks via an uppercase spelling.
    root = _vault_root(tmp_path)
    with pytest.raises(SafetyError, match=r"not found or not readable"):
        resolve_read(root, "metadata/EMBEDDINGS_META.jsonl")


def test_resolve_rejects_traversal_and_absolute(tmp_path: Path) -> None:
    root = _vault_root(tmp_path)
    for bad in ("../outside", "/etc/passwd", "knowledge/../../etc/passwd"):
        with pytest.raises(SafetyError, match=r"escapes the vault root|must be relative to the vault root"):
            resolve_read(root, bad)


def test_resolve_rejects_control_characters(tmp_path: Path) -> None:
    root = _vault_root(tmp_path)
    for bad in ["knowledge/a\nb.md", "knowledge/a\tb.md", "knowledge/a\x00b.md"]:
        with pytest.raises(SafetyError, match=r"control character"):
            resolve_read(root, bad)


def test_resolve_refuses_symlinked_parent(tmp_path: Path) -> None:
    root = _vault_root(tmp_path)
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("s", encoding="utf-8")
    link = root / "knowledge" / "link"
    link.symlink_to(outside)
    # The parent symlink points OUTSIDE the vault, so the escape check
    # fires before the symlink check does — pinned to the guard that
    # actually rejects it rather than the one the test name suggests.
    with pytest.raises(SafetyError, match=r"escapes the vault root"):
        resolve_read(root, "knowledge/link/secret.md")


def test_resolve_inbox_must_stay_under_inbox(tmp_path: Path) -> None:
    root = _vault_root(tmp_path)
    # A '..' escaping inbox/ lands inside the vault but in a ground-truth
    # layer; resolve_inbox must refuse it.
    with pytest.raises(SafetyError, match=r"inbox path must stay under inbox"):
        resolve_inbox(root, "../archive/processed/x")
    # A normal inbox path is fine.
    assert resolve_inbox(root, "docs/new.pdf").name == "new.pdf"


def test_resolve_write_under_allowlist_scope(tmp_path: Path) -> None:
    root = _vault_root(tmp_path)
    assert resolve_write_under_allowlist(root, "knowledge/notes/a.md").name == "a.md"
    # Outside the write allowlist (archive/ is read-only).
    with pytest.raises(SafetyError, match=r"write denied"):
        resolve_write_under_allowlist(root, "archive/processed/x.md")
    # A bare allow-prefix directory is refused (needs a file path).
    with pytest.raises(SafetyError, match=r"refusing to write to bare directory|write denied"):
        resolve_write_under_allowlist(root, "knowledge/notes")


# ------------- concept user-section write can't exceed the note byte cap

def test_update_concept_user_section_refuses_oversized_composed_note(
    env: tuple[Path, ServerConfig, Runtime], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg, runtime = env
    # Shrink the cap so the test stays fast: content passes the per-content
    # check but auto-header + content overflows the composed note.
    patched_max_note_bytes = 200
    monkeypatch.setattr(tools_mod, "MAX_NOTE_BYTES", patched_max_note_bytes)
    slug = "widgets"
    note_rel = f"{CONCEPT_WRITE_PREFIX}/{slug}.md"
    auto = "# Widgets\n\nAuto text.\n\n<!-- AUTO-GENERATED-END -->\n"
    _write(root, note_rel, auto)
    before = (root / note_rel).read_text(encoding="utf-8")

    content = "x" * patched_max_note_bytes  # 200 bytes: passes _check_note_size
    with pytest.raises(ToolError, match="composed concept note would exceed"):
        tool_update_concept_user_section(cfg, runtime, slug=slug, content=content)
    # The note is left untouched — nothing was grown past the cap.
    assert (root / note_rel).read_text(encoding="utf-8") == before


# ------------------------ tool_list never leaks an unreadable dir's path

def test_list_unreadable_dir_returns_generic_error(env: tuple[Path, ServerConfig, Runtime]) -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    root, cfg, runtime = env
    locked = root / "knowledge" / "locked"
    locked.mkdir(parents=True)
    os.chmod(locked, 0o000)
    try:
        with pytest.raises(ToolError, match="not found or not readable"):
            tool_list(cfg, runtime, path="knowledge/locked")
    finally:
        os.chmod(locked, 0o755)  # restore so tmp_path cleanup can recurse


# ------------------------------------------- rate limiting is per-agent

def test_write_rate_limit_is_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh bucket so this test doesn't inherit hits from any other test;
    # monkeypatch restores the module's real bucket at teardown.
    monkeypatch.setattr(
        tools_mod, "_write_bucket", tools_mod._RateBucket(WRITE_RATE_PER_MINUTE)
    )
    token_a = AGENT_VAR.set("agent-a")
    try:
        for _ in range(WRITE_RATE_PER_MINUTE):
            tools_mod._rate_check_write()
        with pytest.raises(ToolError, match="write rate limit exceeded"):
            tools_mod._rate_check_write()
    finally:
        AGENT_VAR.reset(token_a)

    # agent-b's own window is untouched by agent-a exhausting its bucket.
    token_b = AGENT_VAR.set("agent-b")
    try:
        tools_mod._rate_check_write()
    finally:
        AGENT_VAR.reset(token_b)


# ------------------------- AUD-066: the sliding window under a fake clock
# The test above only covers the fresh-bucket branch of one bucket. The
# window logic itself — hits ageing out, the empty-deque eviction, one
# window per key — needs a clock the test controls; sleeping 60 seconds
# is not a test.


class _FakeClock:
    """Hand-advanced stand-in for the ``time`` module.

    ``_RateBucket`` reads the clock as ``tools.time.monotonic()``, and that
    is the only ``time.`` reference in mcp_server.tools, so swapping the
    module attribute pins the 60-second window without a real sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(tools_mod, "time", fake)
    return fake


def test_rate_bucket_evicts_a_key_once_its_window_expires(clock: _FakeClock) -> None:
    # The `del self._hits[key]` branch: a key whose every hit has aged out
    # is dropped, so the map can't keep one entry per agent name ever seen.
    bucket = tools_mod._RateBucket(2)
    assert bucket.allow("agent-a")
    assert bucket.allow("agent-a")
    assert not bucket.allow("agent-a")           # window full
    stale = bucket._hits["agent-a"]
    clock.advance(60.1)                          # every hit is now outside the window
    assert bucket.allow("agent-a")
    # A FRESH deque, not the trimmed old one — the empty one was deleted.
    assert bucket._hits["agent-a"] is not stale
    assert len(bucket._hits["agent-a"]) == 1


def test_rate_bucket_hits_expire_one_at_a_time(clock: _FakeClock) -> None:
    # Sliding window, not a fixed 60-second bucket that empties all at once:
    # each hit frees its own slot exactly 60 seconds after it was taken.
    bucket = tools_mod._RateBucket(2)
    assert bucket.allow("agent-a")               # t = 0
    clock.advance(30.0)
    assert bucket.allow("agent-a")               # t = 30
    assert not bucket.allow("agent-a")
    window = bucket._hits["agent-a"]
    clock.advance(30.1)                          # t = 60.1: only the t=0 hit aged out
    assert bucket.allow("agent-a")               # exactly one slot freed...
    assert not bucket.allow("agent-a")           # ...and only one
    # Partial expiry trims the deque in place; nothing was evicted.
    assert bucket._hits["agent-a"] is window


def test_rate_bucket_windows_are_independent_per_key(clock: _FakeClock) -> None:
    bucket = tools_mod._RateBucket(2)
    assert bucket.allow("agent-a")
    assert bucket.allow("agent-a")
    assert not bucket.allow("agent-a")
    clock.advance(30.0)
    # agent-a's exhausted window is not agent-b's problem.
    assert bucket.allow("agent-b")
    assert bucket.allow("agent-b")
    assert not bucket.allow("agent-b")
    clock.advance(30.1)
    # agent-a's window has rolled off; agent-b's hits are 30.1s old and have not.
    assert bucket.allow("agent-a")
    assert not bucket.allow("agent-b")


def test_search_rate_limit_is_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # The search bucket lives in tools_read but is the same _RateBucket class;
    # a fresh one keeps this test independent of the process-global budget.
    monkeypatch.setattr(
        tools_read_mod,
        "_search_bucket",
        tools_mod._RateBucket(tools_read_mod._SEARCH_RATE_PER_MINUTE),
    )
    token_a = AGENT_VAR.set("agent-a")
    try:
        for _ in range(tools_read_mod._SEARCH_RATE_PER_MINUTE):
            tools_read_mod._rate_check_search()
        with pytest.raises(ToolError, match="search rate limit exceeded"):
            tools_read_mod._rate_check_search()
    finally:
        AGENT_VAR.reset(token_a)

    token_b = AGENT_VAR.set("agent-b")
    try:
        tools_read_mod._rate_check_search()      # agent-b's own window is untouched
    finally:
        AGENT_VAR.reset(token_b)


def test_read_rate_limit_is_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools_read_mod,
        "_read_bucket",
        tools_mod._RateBucket(tools_read_mod._READ_RATE_PER_MINUTE),
    )
    token_a = AGENT_VAR.set("agent-a")
    try:
        for _ in range(tools_read_mod._READ_RATE_PER_MINUTE):
            tools_read_mod._rate_check_read()
        with pytest.raises(ToolError, match="read rate limit exceeded"):
            tools_read_mod._rate_check_read()
    finally:
        AGENT_VAR.reset(token_a)

    token_b = AGENT_VAR.set("agent-b")
    try:
        tools_read_mod._rate_check_read()
    finally:
        AGENT_VAR.reset(token_b)


def test_callers_without_an_identity_share_one_bucket() -> None:
    # Per-agent windows are keyed on current_agent(), which is the fail-safe
    # "unknown" for anything that never passed the auth middleware. Two such
    # callers must NOT get a budget each — that would be a free bucket for
    # every unauthenticated path. A brand-new contextvars.Context is empty,
    # so AGENT_VAR falls back to its default exactly as it would there.
    bucket = tools_mod._RateBucket(3)
    first, second = contextvars.Context(), contextvars.Context()

    def take() -> bool:
        return bucket.allow(current_agent())

    for _ in range(3):
        assert first.run(take)
    assert not second.run(take)          # same bucket, already exhausted
    assert list(bucket._hits) == ["unknown"]
