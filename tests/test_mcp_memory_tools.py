"""Tests for the memory MCP tools (B1b): memory_search + profile_update.

memory_search is exercised against a monkeypatched
``ingest_lib.recency.memory_search`` stub — the tool's own job is input
validation, ValueError mapping, the read gate, and field mapping; the
ranking maths is covered in test_recency. profile_update runs end-to-end
over a throwaway git vault. Fully offline.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ingest_lib import recency
from ingest_lib.knowledge import KNOWLEDGE_EXTRACTOR
from mcp_server import tools as tools_mod
from mcp_server.audit import AuditLog
from mcp_server.config import PROFILE_NOTE_PATH, ServerConfig
from mcp_server.identity import AGENT_VAR
from mcp_server.memory_tools import tool_memory_search, tool_profile_update
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import (
    ToolError,
    tool_append_to_note,
    tool_create_note,
    tool_replace_note,
)


# --------------------------------------------------------------- harness

def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _make_vault(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    return root


def _cfg(root: Path, profile_max_bytes: int = 4096) -> ServerConfig:
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
        profile_max_bytes=profile_max_bytes,
    )


def _runtime(root: Path) -> Runtime:
    audit = AuditLog(root)
    return Runtime(
        audit=audit,
        push_worker=PushWorker(root, remote="origin", branch="main", enabled=False),
        refresher=IndexRefresher(root, audit=audit, enabled=False),
    )


@pytest.fixture(autouse=True)
def _roomy_rate_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Module-global buckets are shared across the whole pytest process;
    # fresh roomy ones keep tests order-independent.
    monkeypatch.setattr(tools_mod, "_write_bucket", tools_mod._RateBucket(10_000))
    monkeypatch.setattr(tools_mod, "_search_bucket", tools_mod._RateBucket(10_000))


@pytest.fixture()
def env(tmp_path: Path):
    root = _make_vault(tmp_path)
    token = AGENT_VAR.set("agent-a")
    try:
        yield root, _cfg(root), _runtime(root)
    finally:
        AGENT_VAR.reset(token)


def _hit(path: str, *, origin: str = KNOWLEDGE_EXTRACTOR, score: float = 0.5) -> recency.MemoryHit:
    return recency.MemoryHit(
        score=score, cosine=0.9, recency=0.6, status_weight=1.0,
        source_relative_path=path, title="T", snippet="snip",
        origin=origin, updated="2026-06-01", chunk_idx=2,
    )


def _access_rows(root: Path) -> list[dict]:
    # Reads/searches land in the ACCESS stream (writes go to mcp-audit.jsonl).
    path = root / "logs" / "mcp-access.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------- memory_search

def test_memory_search_maps_stub_hits(env, monkeypatch: pytest.MonkeyPatch) -> None:
    root, cfg, runtime = env
    captured: dict = {}

    def fake(paths, query, *, top_k, halflife_days, types, logger=None, now=None):
        captured.update(
            root=paths.root, query=query, top_k=top_k,
            halflife_days=halflife_days, types=types,
        )
        return [_hit("knowledge/people/anna.md")]

    monkeypatch.setattr(recency, "memory_search", fake)
    out = tool_memory_search(
        cfg, runtime, query="anna", top_k=5,
        recency_halflife_days=7.0, types=["people"],
    )
    assert captured == {
        "root": root, "query": "anna", "top_k": 5,
        "halflife_days": 7.0, "types": ["people"],
    }
    assert len(out.hits) == 1
    h = out.hits[0]
    assert h.score == 0.5 and h.cosine == 0.9 and h.recency == 0.6
    assert h.status_weight == 1.0
    assert h.source_relative_path == "knowledge/people/anna.md"
    assert h.title == "T" and h.snippet == "snip"
    assert h.updated == "2026-06-01" and h.chunk_idx == 2
    # The access row records the query and the gated hit paths.
    row = _access_rows(root)[-1]
    assert row["tool"] == "memory_search"
    assert row["agent"] == "agent-a"
    assert row["query"] == "anna"
    assert row["paths"] == ["knowledge/people/anna.md"]


def test_memory_search_unknown_type_becomes_tool_error(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, cfg, runtime = env

    def fake(paths, query, **kwargs):
        raise ValueError("unknown type token(s) ['bogus'] — valid: people, ...")

    monkeypatch.setattr(recency, "memory_search", fake)
    with pytest.raises(ToolError, match="bogus"):
        tool_memory_search(cfg, runtime, query="anna", types=["bogus"])


def test_memory_search_read_gate_drops_denied_paths(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg, runtime = env

    def fake(paths, query, **kwargs):
        return [
            _hit("knowledge/people/anna.md"),
            # A "knowledge note" outside the read allowlist: the gate path
            # is the hit itself, and logs/ is never readable.
            _hit("logs/private.md"),
        ]

    monkeypatch.setattr(recency, "memory_search", fake)
    out = tool_memory_search(cfg, runtime, query="anna")
    assert [h.source_relative_path for h in out.hits] == ["knowledge/people/anna.md"]
    # The denied hit never reaches the access trail either.
    assert _access_rows(root)[-1]["paths"] == ["knowledge/people/anna.md"]


@pytest.mark.parametrize("kwargs,expected", [
    ({"query": "  "}, "non-empty"),
    ({"query": "x", "top_k": 0}, r"\[1, 50\]"),
    ({"query": "x", "top_k": 51}, r"\[1, 50\]"),
    ({"query": "x", "recency_halflife_days": 0.5}, r"\[1, 3650\]"),
    ({"query": "x", "recency_halflife_days": 4000.0}, r"\[1, 3650\]"),
])
def test_memory_search_validates_inputs(env, kwargs: dict, expected: str) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match=expected):
        tool_memory_search(cfg, runtime, **kwargs)


# ---------------------------------------------------------- profile_update

def test_profile_update_creates_then_replaces(env) -> None:
    root, cfg, runtime = env
    res = tool_profile_update(cfg, runtime, content="# Profile\n\nLikes pnpm.\n")
    assert res.path == PROFILE_NOTE_PATH
    assert res.committed and res.commit_sha
    on_disk = (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8")
    assert "last_written_by: 'agent:agent-a'" in on_disk
    assert "written_via: mcp" in on_disk
    assert "memory_status: consolidated" in on_disk
    assert on_disk.endswith("Likes pnpm.\n")
    assert _git(root, "log", "-1", "--format=%s") == "mcp(agent-a): profile update"

    # Replace: same tool, full rewrite — old content is gone.
    res2 = tool_profile_update(cfg, runtime, content="# Profile\n\nPrefers uv.\n")
    assert res2.committed and res2.commit_sha != res.commit_sha
    replaced = (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8")
    assert "Prefers uv." in replaced
    assert "Likes pnpm." not in replaced
    assert "memory_status: consolidated" in replaced


def test_profile_update_overrides_client_memory_status(env) -> None:
    root, cfg, runtime = env
    content = (
        "---\n"
        "title: 'Assistant Profile'\n"
        "memory_status: unconsolidated\n"
        "---\n"
        "# Profile\n"
    )
    tool_profile_update(cfg, runtime, content=content)
    on_disk = (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8")
    assert "memory_status: consolidated" in on_disk
    assert "memory_status: unconsolidated" not in on_disk
    assert "title: 'Assistant Profile'" in on_disk  # user keys survive


def test_profile_update_refuses_over_budget(env) -> None:
    root, cfg, runtime = env
    big = "x" * (cfg.profile_max_bytes + 1)
    with pytest.raises(ToolError) as exc:
        tool_profile_update(cfg, runtime, content=big)
    msg = str(exc.value)
    assert str(cfg.profile_max_bytes) in msg
    assert "the profile is a token budget, not a notebook: curate, don't accumulate" in msg
    assert not (root / PROFILE_NOTE_PATH).exists()  # nothing was written


def test_profile_update_budget_counts_provenance_overhead(env) -> None:
    # F078: the budget is enforced on the FINAL note (content + provenance +
    # memory_status), so content that fits only before stamping is refused.
    root, cfg, runtime = env
    content = "x" * cfg.profile_max_bytes  # fits pre-stamp, overflows after
    with pytest.raises(ToolError, match="after provenance stamping"):
        tool_profile_update(cfg, runtime, content=content)
    assert not (root / PROFILE_NOTE_PATH).exists()


def test_profile_update_accepts_within_budget(env) -> None:
    root, cfg, runtime = env
    # Leave headroom for the stamped frontmatter (~90 bytes).
    content = "x" * (cfg.profile_max_bytes - 512)
    res = tool_profile_update(cfg, runtime, content=content)
    assert res.committed
    assert (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8").endswith(content)
    assert len((root / PROFILE_NOTE_PATH).read_text(encoding="utf-8").encode()) <= cfg.profile_max_bytes


# ----------------------------------------- PROFILE.md budget can't be bypassed

@pytest.mark.parametrize("write_tool", [tool_create_note, tool_replace_note, tool_append_to_note])
def test_general_write_verbs_refuse_profile_path(env, write_tool) -> None:
    # PROFILE.md is inside the write allowlist (knowledge/assistant/), so the
    # general note verbs would otherwise accept it and dodge the byte budget
    # profile_update enforces. They must refuse it and point at profile_update.
    root, cfg, runtime = env
    with pytest.raises(ToolError, match="profile_update"):
        write_tool(cfg, runtime, path=PROFILE_NOTE_PATH, content="# Profile\n\nbypass\n")
    assert not (root / PROFILE_NOTE_PATH).exists()  # nothing was written


def test_general_write_verbs_refuse_profile_even_when_it_exists(env) -> None:
    # profile_update is still the only door even once the file exists: the
    # refusal fires before the replace tool's exists() check.
    root, cfg, runtime = env
    tool_profile_update(cfg, runtime, content="# Profile\n\nLikes uv.\n")
    before = (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8")
    with pytest.raises(ToolError, match="byte-budgeted"):
        tool_replace_note(cfg, runtime, path=PROFILE_NOTE_PATH, content="# Profile\n\nclobbered\n")
    # The profile is untouched, and profile_update itself still works.
    assert (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8") == before
    res = tool_profile_update(cfg, runtime, content="# Profile\n\nNow prefers pnpm.\n")
    assert res.committed
    assert "pnpm" in (root / PROFILE_NOTE_PATH).read_text(encoding="utf-8")
