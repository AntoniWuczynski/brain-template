"""vault_chunk_context: expand a search hit into its neighbouring chunks."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import ToolError, tool_chunk_context


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


def _seed_index(root: Path, rows: list[dict]) -> None:
    meta = root / "metadata" / "embeddings_meta.jsonl"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _rows(src: str, n: int, origin: str = "knowledge-note") -> list[dict]:
    return [
        {"source_relative_path": src, "text": f"chunk {i}", "chunk_idx": i,
         "title": src, "origin": origin, "source_hash": "h"}
        for i in range(n)
    ]


def test_chunk_context_returns_window_around_target(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed_index(root, _rows("knowledge/notes/a.md", 5))
    out = tool_chunk_context(
        _cfg(root), _runtime(root), "knowledge/notes/a.md", chunk_idx=2,
        before=1, after=1,
    )
    assert out.total_chunks == 5
    assert [c.chunk_idx for c in out.chunks] == [1, 2, 3]
    assert [c.is_target for c in out.chunks] == [False, True, False]
    assert out.chunks[1].text == "chunk 2"


def test_chunk_context_clamps_at_edges(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed_index(root, _rows("knowledge/notes/a.md", 3))
    out = tool_chunk_context(
        _cfg(root), _runtime(root), "knowledge/notes/a.md", chunk_idx=0,
        before=2, after=2,
    )
    assert [c.chunk_idx for c in out.chunks] == [0, 1, 2]  # no negative indices


def test_chunk_context_unknown_source_errors(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed_index(root, _rows("knowledge/notes/a.md", 2))
    with pytest.raises(ToolError, match="no indexed chunks"):
        tool_chunk_context(_cfg(root), _runtime(root), "knowledge/notes/missing.md", 0)


def test_chunk_context_gated_source_is_refused(tmp_path: Path) -> None:
    # A source whose backing artifact isn't readable under the policy (e.g.
    # a secrets path) must be refused, mirroring vault_search's gate.
    root = _make_vault(tmp_path)
    _seed_index(root, _rows(".env", 3, origin="knowledge-note"))
    with pytest.raises(ToolError, match="not found or not readable"):
        tool_chunk_context(_cfg(root), _runtime(root), ".env", 1)


def test_chunk_context_validates_args(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    _seed_index(root, _rows("knowledge/notes/a.md", 2))
    cfg, rt = _cfg(root), _runtime(root)
    with pytest.raises(ToolError, match="chunk_idx"):
        tool_chunk_context(cfg, rt, "knowledge/notes/a.md", -1)
    with pytest.raises(ToolError, match="before/after"):
        tool_chunk_context(cfg, rt, "knowledge/notes/a.md", 0, before=99)
