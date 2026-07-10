"""Security-boundary regression tests for the MCP write/read surface.

These pin the fixes for the traversal, existence-oracle, profile-budget-
bypass, and query-DoS findings. They drive the real tools over a throwaway
git vault with background workers disabled — no live server, fully offline.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ingest_lib.relations import is_valid_node_id  # type: ignore[import-not-found]

from mcp_server import tools as tools_mod
from mcp_server.audit import AuditLog
from mcp_server.safety import (
    SafetyError,
    resolve_inbox,
    resolve_read,
    resolve_write_under_allowlist,
)
from mcp_server.config import CONCEPT_WRITE_PREFIX, ServerConfig
from mcp_server.entity_tools import (
    tool_entity_append_fact,
    tool_entity_upsert_relation,
    tool_meeting_create,
)
from mcp_server.identity import AGENT_VAR
from mcp_server.memory_tools import tool_profile_update
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import (
    MAX_QUERY_CHARS,
    ToolError,
    tool_list,
    tool_related,
    tool_search,
    tool_update_concept_user_section,
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
def env(tmp_path: Path):
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
def test_valid_node_ids_accepted(good) -> None:
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
def test_invalid_node_ids_rejected(bad) -> None:
    assert not is_valid_node_id(bad)


# ---------------------------------- F000: meeting_create traversal blocked

def test_meeting_create_rejects_traversal_attendee(env) -> None:
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

def test_upsert_relation_rejects_traversal_target(env) -> None:
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

def test_append_fact_source_oracle_collapsed(env) -> None:
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

def test_entity_append_fact_refuses_profile(env) -> None:
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

def test_search_rejects_oversized_query(env) -> None:
    root, cfg, runtime = env
    with pytest.raises(ToolError, match="query is"):
        tool_search(cfg, runtime, query="x" * (MAX_QUERY_CHARS + 1))


def test_related_rejects_oversized_concept(env) -> None:
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


def test_resolve_read_allows_and_denies(tmp_path):
    root = _vault_root(tmp_path)
    (root / "knowledge" / "notes" / "a.md").write_text("x", encoding="utf-8")
    # Allowed prefix + root doc file.
    assert resolve_read(root, "knowledge/notes/a.md").name == "a.md"
    (root / "README.md").write_text("x", encoding="utf-8")
    assert resolve_read(root, "README.md").name == "README.md"
    # Outside the read allowlist (server code, scripts/).
    with pytest.raises(SafetyError):
        resolve_read(root, "mcp_server/tools.py")
    # Deny-listed within an allowed area (logs, .env, the embeddings index).
    for denied in ("logs/x.log", ".env", "metadata/embeddings_meta.jsonl"):
        with pytest.raises(SafetyError):
            resolve_read(root, denied)


def test_resolve_read_deny_is_casefolded(tmp_path):
    # On a case-insensitive FS the DENY check must casefold, or the
    # embeddings text leaks via an uppercase spelling.
    root = _vault_root(tmp_path)
    with pytest.raises(SafetyError):
        resolve_read(root, "metadata/EMBEDDINGS_META.jsonl")


def test_resolve_rejects_traversal_and_absolute(tmp_path):
    root = _vault_root(tmp_path)
    for bad in ("../outside", "/etc/passwd", "knowledge/../../etc/passwd"):
        with pytest.raises(SafetyError):
            resolve_read(root, bad)


def test_resolve_rejects_control_characters(tmp_path):
    root = _vault_root(tmp_path)
    for bad in ["knowledge/a\nb.md", "knowledge/a\tb.md", "knowledge/a\x00b.md"]:
        with pytest.raises(SafetyError):
            resolve_read(root, bad)


def test_resolve_refuses_symlinked_parent(tmp_path):
    root = _vault_root(tmp_path)
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("s", encoding="utf-8")
    link = root / "knowledge" / "link"
    link.symlink_to(outside)
    with pytest.raises(SafetyError):
        resolve_read(root, "knowledge/link/secret.md")


def test_resolve_inbox_must_stay_under_inbox(tmp_path):
    root = _vault_root(tmp_path)
    # A '..' escaping inbox/ lands inside the vault but in a ground-truth
    # layer; resolve_inbox must refuse it.
    with pytest.raises(SafetyError):
        resolve_inbox(root, "../archive/processed/x")
    # A normal inbox path is fine.
    assert resolve_inbox(root, "docs/new.pdf").name == "new.pdf"


def test_resolve_write_under_allowlist_scope(tmp_path):
    root = _vault_root(tmp_path)
    assert resolve_write_under_allowlist(root, "knowledge/notes/a.md").name == "a.md"
    # Outside the write allowlist (archive/ is read-only).
    with pytest.raises(SafetyError):
        resolve_write_under_allowlist(root, "archive/processed/x.md")
    # A bare allow-prefix directory is refused (needs a file path).
    with pytest.raises(SafetyError):
        resolve_write_under_allowlist(root, "knowledge/notes")


# ------------- concept user-section write can't exceed the note byte cap

def test_update_concept_user_section_refuses_oversized_composed_note(
    env, monkeypatch
) -> None:
    root, cfg, runtime = env
    # Shrink the cap so the test stays fast: content passes the per-content
    # check but auto-header + content overflows the composed note.
    monkeypatch.setattr(tools_mod, "MAX_NOTE_BYTES", 200)
    slug = "widgets"
    note_rel = f"{CONCEPT_WRITE_PREFIX}/{slug}.md"
    auto = "# Widgets\n\nAuto text.\n\n<!-- AUTO-GENERATED-END -->\n"
    _write(root, note_rel, auto)
    before = (root / note_rel).read_text(encoding="utf-8")

    content = "x" * tools_mod.MAX_NOTE_BYTES  # 200 bytes: passes _check_note_size
    with pytest.raises(ToolError, match="composed concept note would exceed"):
        tool_update_concept_user_section(cfg, runtime, slug=slug, content=content)
    # The note is left untouched — nothing was grown past the cap.
    assert (root / note_rel).read_text(encoding="utf-8") == before


# ------------------------ tool_list never leaks an unreadable dir's path

def test_list_unreadable_dir_returns_generic_error(env) -> None:
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
