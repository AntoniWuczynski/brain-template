"""Write-policy and search-gate coverage for the MCP tools.

These invariants previously lived only in throwaway ``python -m`` scripts
(mcp_server/test_replace_note.py, mcp_server/test_search_gate.py) that CI
never ran — so when derived_note_relpath changed to keep the source
extension, the gate script's expectations silently went stale. Ported here
as real pytest cases over a throwaway git vault so they run every CI push.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mcp_server.audit import AuditLog
from mcp_server.config import MAX_NOTE_BYTES, ServerConfig
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.safety import SafetyError
from mcp_server.runtime import Runtime
from mcp_server.tools import (
    ToolError,
    _hit_gate_path,
    tool_create_note,
    tool_replace_note,
)


@pytest.fixture(autouse=True)
def _roomy_rate_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_server.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_write_bucket", tools_mod._RateBucket(10_000))


def _make_vault(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
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


# ------------------------------------------------------- search-hit read gate

def test_source_hit_gates_on_processed_twin() -> None:
    # An ingested source's content is only returned if its processed artifact
    # is readable — gate on archive/processed/<rel>.<ext>.md (the twin keeps
    # the source extension since the D7 collision fix).
    assert _hit_gate_path(
        "university/COMP0023/04_error_coding.pdf", "pdf-mineru"
    ) == "archive/processed/university/COMP0023/04_error_coding.pdf.md"


def test_knowledge_note_hit_gates_on_its_own_path() -> None:
    assert _hit_gate_path(
        "knowledge/projects/brain/brain.md", "knowledge-note"
    ) == "knowledge/projects/brain/brain.md"


def test_ingested_source_labelled_knowledge_gates_on_twin() -> None:
    # A source dropped at inbox/knowledge/x.pdf is LABELLED knowledge/ but is
    # NOT a vault note — origin, not the path prefix, decides.
    assert _hit_gate_path(
        "knowledge/x.pdf", "pdf-pypdf-fallback"
    ) == "archive/processed/knowledge/x.pdf.md"


def test_legacy_row_without_origin_falls_back_to_prefix() -> None:
    # Rows built before `origin` existed: a knowledge/ path gates on itself,
    # a non-knowledge path gates on its processed twin.
    assert _hit_gate_path("knowledge/projects/brain/brain.md", "") == (
        "knowledge/projects/brain/brain.md"
    )
    assert _hit_gate_path("uni/x.pdf", "") == "archive/processed/uni/x.pdf.md"


# ------------------------------------------------------------- write policy

def test_replace_refuses_oversize_content(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    cfg, runtime = _cfg(root), _runtime(root)
    note = "knowledge/notes/x.md"
    tool_create_note(cfg, runtime, path=note, content="# x\n")
    with pytest.raises((ToolError, SafetyError)) as exc:
        tool_replace_note(cfg, runtime, path=note, content="x" * (MAX_NOTE_BYTES + 1))
    assert "max" in str(exc.value).lower()


def test_replace_refuses_nonexistent_note(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    cfg, runtime = _cfg(root), _runtime(root)
    with pytest.raises(ToolError) as exc:
        tool_replace_note(cfg, runtime, path="knowledge/notes/absent.md", content="x")
    assert "does not exist" in str(exc.value).lower()


def test_note_verbs_refuse_inbox_paths(tmp_path: Path) -> None:
    # inbox/ is writable only via drop_inbox_file; the note verbs must refuse
    # it (WRITE_ALLOW_PREFIXES excludes inbox/) and leave a pending source
    # byte-identical — replace would otherwise be the one tool able to
    # destroy an uncommitted inbox file.
    root = _make_vault(tmp_path)
    cfg, runtime = _cfg(root), _runtime(root)
    (root / "inbox").mkdir()
    pending = root / "inbox" / "pending.md"
    pending.write_text("original source\n", encoding="utf-8")

    with pytest.raises((ToolError, SafetyError)) as exc:
        tool_replace_note(cfg, runtime, path="inbox/pending.md", content="DESTROYED")
    assert "denied" in str(exc.value).lower()
    assert pending.read_text(encoding="utf-8") == "original source\n"

    with pytest.raises((ToolError, SafetyError)) as exc2:
        tool_create_note(cfg, runtime, path="inbox/new.md", content="nope")
    assert "denied" in str(exc2.value).lower()
