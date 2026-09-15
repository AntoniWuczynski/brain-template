"""Tests for the ``mcp_server.app`` adapter layer and ``runtime.build_runtime``
(F4): the layer that binds tool functions to FastMCP was never exercised by
the suite at all — AUD-014's happy-path tests stop at the tool-function
layer (``mcp_server.tools``/``tools_read``), never through ``_register_tools``.

Builds a real ``FastMCP`` instance against a throwaway git vault (same
harness as tests/test_mcp_happy_path.py) and drives it through
``list_tools``/``call_tool``, plus a direct check of how ``build_runtime``
wires the push worker and index refresher from ``ServerConfig``. No model
load, no network.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock

from ingest_lib import summarize as _summarize
from mcp_server import app as _app
from mcp_server.app import _register_tools, build_app, unavailable_extractor_families
from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.identity import AGENT_VAR
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime, build_runtime

# The four env vars ingest_lib.summarize._select_provider auto-detects a
# provider from — cleared in every extractor-availability test below so the
# "vision" check (AUD-124) is deterministic regardless of the machine
# running the suite.
_LLM_PROVIDER_ENV = (
    "BRAIN_LLM_PROVIDER", "BRAIN_SKIP_SUMMARY", "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "BRAIN_LOCAL_URL",
)


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
    (root / "README.md").write_text("# hi\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _cfg(root: Path) -> ServerConfig:
    return ServerConfig(
        vault_root=root,
        tokens=(("x" * 24, "default"),),
        bind_host="127.0.0.1", bind_port=0, git_push_on_write=False,
        git_remote="origin", git_branch="main", log_level="warning",
        allowed_hosts=(), profile_max_bytes=4096,
    )


async def _call_tool(
    mcp: FastMCP, name: str, arguments: dict[str, object]
) -> tuple[Sequence[ContentBlock], dict[str, object]]:
    """``FastMCP.call_tool``'s declared return type
    (``Sequence[ContentBlock] | dict[str, Any]``) is stale for the
    installed SDK version: verified at runtime (scratchpad recipe this
    test is based on) it returns a ``(content, structured)`` tuple for
    any tool with an output schema — every tool registered here has one.
    Cast to that real runtime shape rather than typing around it."""
    raw = await mcp.call_tool(name, arguments)
    return cast("tuple[Sequence[ContentBlock], dict[str, object]]", raw)


def _runtime(root: Path) -> Runtime:
    audit = AuditLog(root)
    return Runtime(
        audit=audit,
        push_worker=PushWorker(root, remote="origin", branch="main", enabled=False),
        refresher=IndexRefresher(root, audit=audit, enabled=False),
    )


_EXPECTED_TOOL_NAMES = frozenset({
    "vault_search", "vault_read", "vault_chunk_context", "vault_list",
    "vault_metadata_query", "vault_related", "vault_create_note",
    "vault_replace_note", "vault_append_to_note",
    "vault_update_concept_user_section", "vault_update_compiled_truth",
    "vault_drop_inbox_file",
    "entity_upsert_relation", "entity_append_fact", "relations_query",
    "meeting_create", "memory_search", "profile_update",
})


def test_register_tools_lists_every_tool_with_an_output_schema(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    mcp = FastMCP("brain-vault-test")
    _register_tools(mcp, _cfg(root), _runtime(root))

    tools = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in tools}

    assert set(by_name) == _EXPECTED_TOOL_NAMES
    assert len(by_name) == 18
    for name, tool in by_name.items():
        assert tool.outputSchema, f"{name} has no outputSchema"


def test_call_tool_vault_read_returns_structured_content_matching_readout(
    tmp_path: Path,
) -> None:
    root = _make_vault(tmp_path)
    mcp = FastMCP("brain-vault-test")
    _register_tools(mcp, _cfg(root), _runtime(root))

    token = AGENT_VAR.set("default")
    try:
        content, structured = asyncio.run(_call_tool(mcp, "vault_read", {"path": "README.md"}))
    finally:
        AGENT_VAR.reset(token)

    assert content and content[0].type == "text"
    assert structured == {
        "path": "README.md",
        "content": "# hi\n",
        "size_bytes": 5,
    }


def test_call_tool_vault_create_note_commits_through_the_adapter(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    mcp = FastMCP("brain-vault-test")
    _register_tools(mcp, _cfg(root), _runtime(root))

    token = AGENT_VAR.set("default")
    try:
        _content, structured = asyncio.run(_call_tool(
            mcp,
            "vault_create_note",
            {
                "path": "knowledge/notes/x.md",
                "content": "---\ntitle: x\ntype: note\n---\n\n# x\n",
            },
        ))
    finally:
        AGENT_VAR.reset(token)

    assert structured["committed"] is True
    assert structured["commit_sha"]
    assert (root / "knowledge/notes/x.md").is_file()
    log_line = _git(root, "log", "-1", "--format=%s")
    assert log_line == "mcp(default): create note knowledge/notes/x.md"


def test_call_tool_unknown_tool_name_raises(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    mcp = FastMCP("brain-vault-test")
    _register_tools(mcp, _cfg(root), _runtime(root))

    with pytest.raises(Exception):  # noqa: B017 — FastMCP's own ToolError type
        asyncio.run(mcp.call_tool("not_a_real_tool", {}))


def test_build_runtime_wires_push_worker_and_refresher_from_config(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    cfg = ServerConfig(
        vault_root=root,
        tokens=(("x" * 24, "default"),),
        bind_host="127.0.0.1", bind_port=0,
        git_push_on_write=True,
        git_remote="upstream", git_branch="trunk", log_level="warning",
        allowed_hosts=(), profile_max_bytes=4096,
    )

    runtime = build_runtime(cfg)
    try:
        assert runtime.push_worker.vault_root == root
        assert runtime.push_worker.branch == "trunk"
        assert runtime.push_worker._remote == "upstream"
        assert runtime.push_worker._enabled is True
        assert runtime.refresher._branch == "trunk"
        # The refresher's derived-notes commits ride the push worker's
        # own coalescing queue (its request_push), not a separate path.
        assert runtime.refresher._request_push == runtime.push_worker.request_push
    finally:
        runtime.refresher.stop()
        runtime.push_worker.stop()


# --------------------------------------------------- extractor availability (AUD-124)


def test_unavailable_extractor_families_reports_all_three_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_app, "_mineru_available", lambda: False)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    for key in _LLM_PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)

    missing = unavailable_extractor_families()

    assert len(missing) == 3
    assert any("MinerU" in m for m in missing)
    assert any("whisper" in m for m in missing)
    assert any("vision" in m for m in missing)


def test_unavailable_extractor_families_empty_when_everything_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_app, "_mineru_available", lambda: True)
    monkeypatch.setattr(_summarize, "_select_provider", lambda: "anthropic")
    fake_module = type(sys)("faster_whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    assert unavailable_extractor_families() == ()


def test_unavailable_extractor_families_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_app, "_mineru_available", lambda: True)
    monkeypatch.setattr(_summarize, "_select_provider", lambda: None)
    fake_module = type(sys)("faster_whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    missing = unavailable_extractor_families()

    assert len(missing) == 1
    assert "vision" in missing[0]


def test_unavailable_extractor_families_vision_available_even_with_skip_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BRAIN_SKIP_SUMMARY=1 disables the summarizer, not the VLM extractor:
    vlm.py gates on ``_select_provider()`` directly (finding #2), so the
    boot-time vision check must track that gate, not ``is_enabled()``'s
    broader one which also trips on BRAIN_SKIP_SUMMARY."""
    monkeypatch.setattr(_app, "_mineru_available", lambda: True)
    monkeypatch.setattr(_summarize, "_select_provider", lambda: "anthropic")
    monkeypatch.setenv("BRAIN_SKIP_SUMMARY", "1")
    fake_module = type(sys)("faster_whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    assert unavailable_extractor_families() == ()


def test_build_app_logs_which_extractor_families_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    root = _make_vault(tmp_path)
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(root))
    monkeypatch.setenv("BRAIN_MCP_BEARER_TOKEN", "x" * 40)
    monkeypatch.setenv("BRAIN_MCP_GIT_PUSH_ON_WRITE", "0")
    monkeypatch.setenv("BRAIN_MCP_LOG_LEVEL", "info")
    monkeypatch.setattr(_app, "_mineru_available", lambda: False)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    for key in _LLM_PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)

    with caplog.at_level(logging.WARNING, logger="mcp_server.app"):
        build_app()  # construction only — no threads start without a write/search

    assert any(
        "extractor families unavailable" in r.message and "MinerU" in r.message
        for r in caplog.records
    )
