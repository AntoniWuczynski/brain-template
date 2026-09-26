"""vault_outline: a source's table of contents built from its indexed chunks."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.errors import ToolError
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools_read import tool_outline


def _make_vault(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    return root


def _cfg(root: Path) -> ServerConfig:
    return ServerConfig(
        vault_root=root, tokens=(("x" * 24, "default"),),
        bind_host="127.0.0.1", bind_port=0, git_push_on_write=False,
        git_remote="origin", git_branch="main", log_level="warning",
        allowed_hosts=(), profile_max_bytes=4096,
    )


def _runtime(root: Path) -> Runtime:
    audit = AuditLog(root)
    return Runtime(
        audit=audit,
        push_worker=PushWorker(root, remote="origin", branch="main", enabled=False),
        refresher=IndexRefresher(root, audit=audit, enabled=False),
    )


def _seed(root: Path, src: str, headings: list[str], origin: str = "knowledge-note") -> None:
    meta = root / "metadata" / "embeddings_meta.jsonl"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("w", encoding="utf-8") as fh:
        for i, h in enumerate(headings):
            fh.write(json.dumps({
                "source_relative_path": src, "text": "x" * (i + 1), "chunk_idx": i,
                "title": "Graphs", "origin": origin, "source_hash": "h", "heading_path": h,
            }) + "\n")


def test_outline_groups_consecutive_chunks_by_heading(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed(root, "knowledge/notes/g.md", ["", "4 Graphs", "4 Graphs", "4 Graphs > 4.1 Undirected",
                                          "4 Graphs"])
    out = tool_outline(_cfg(root), _runtime(root), "knowledge/notes/g.md")
    assert out.title == "Graphs"
    assert out.total_chunks == 5
    assert [(s.heading_path, s.first_chunk, s.last_chunk, s.chars) for s in out.sections] == [
        ("", 0, 0, 1),
        ("4 Graphs", 1, 2, 5),
        ("4 Graphs > 4.1 Undirected", 3, 3, 4),
        ("4 Graphs", 4, 4, 5),
    ]


def test_outline_unknown_source_errors(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed(root, "knowledge/notes/g.md", ["a"])
    with pytest.raises(ToolError, match="no indexed chunks"):
        tool_outline(_cfg(root), _runtime(root), "knowledge/notes/missing.md")


def test_outline_gated_source_is_refused(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed(root, ".env", ["a"])
    with pytest.raises(ToolError, match="not found or not readable"):
        tool_outline(_cfg(root), _runtime(root), ".env")


def test_outline_rejects_empty_path(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    with pytest.raises(ToolError, match="non-empty"):
        tool_outline(_cfg(root), _runtime(root), " ")
