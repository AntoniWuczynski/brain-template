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
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock

from mcp_server.app import _register_tools
from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.identity import AGENT_VAR
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime, build_runtime


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
