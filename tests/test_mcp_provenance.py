"""Tests for mcp_server.provenance and the rewired MCP write path.

Two layers:

- Pure-text checks for ``stamp_provenance`` (mode x frontmatter-presence
  x spoofed-key matrix, byte-exact body preservation) and
  ``frontmatter_signature`` (graph-input change detection).
- An end-to-end pass of the write tools over a throwaway git vault with
  a real Runtime whose background workers are disabled — asserting the
  on-disk stamps, the ``mcp(<agent>)`` commit messages, the new
  WriteResult fields, and the audit JSONL rows. Fully offline.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.identity import AGENT_VAR
from mcp_server.provenance import frontmatter_signature, stamp_provenance
from mcp_server import tools as tools_mod
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import ToolError, tool_create_note, tool_replace_note


@pytest.fixture(autouse=True)
def _roomy_rate_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    # The write bucket (30/min) is module-global and shared across the whole
    # pytest process; without a fresh roomy bucket per test this file's write
    # calls would eat the shared budget and make sibling tests order-dependent
    # (the other MCP test files already carry this fixture).
    monkeypatch.setattr(tools_mod, "_write_bucket", tools_mod._RateBucket(10_000))


# ------------------------------------------------------------ stamping

_BODY = "# Anna\n\nBody stays byte-for-byte.\n\n---\n\nEven a fake fence in it.\n"

_WITH_FM = (
    "---\n"
    "title: 'Anna Kowalska'\n"
    "type: person\n"
    "topics: [People]\n"
    "---\n" + _BODY
)

_SPOOFED = (
    "---\n"
    "title: x\n"
    "author: 'agent:someone-else'\n"
    "written_via: telepathy\n"
    "memory_status: consolidated\n"
    "---\n" + _BODY
)


def _fm_lines(text: str) -> list[str]:
    assert text.startswith("---\n")
    return text[4:].split("\n---\n", 1)[0].splitlines()


def _body_of(text: str) -> str:
    return text.split("\n---\n", 1)[1]


def test_create_without_frontmatter_prepends_minimal_block() -> None:
    out = stamp_provenance(_BODY, agent="agent-a", mode="create", memory_area=False)
    assert _fm_lines(out) == ["author: 'agent:agent-a'", "written_via: mcp"]
    # The whole original content — including its decoy '---' — is the body.
    assert out == "---\nauthor: 'agent:agent-a'\nwritten_via: mcp\n---\n" + _BODY


def test_create_with_frontmatter_keeps_user_keys_and_body() -> None:
    out = stamp_provenance(_WITH_FM, agent="agent-a", mode="create", memory_area=False)
    lines = _fm_lines(out)
    # User keys preserved verbatim and first; provenance appended before the fence.
    assert lines[:3] == ["title: 'Anna Kowalska'", "type: person", "topics: [People]"]
    assert "author: 'agent:agent-a'" in lines
    assert "written_via: mcp" in lines
    assert "memory_status" not in out  # not a memory area
    assert _body_of(out) == _BODY  # byte-for-byte


def test_create_in_memory_area_adds_unconsolidated_status() -> None:
    out = stamp_provenance(_WITH_FM, agent="agent-a", mode="create", memory_area=True)
    assert "memory_status: unconsolidated" in _fm_lines(out)


def test_create_overrides_client_spoofed_provenance() -> None:
    out = stamp_provenance(_SPOOFED, agent="agent-a", mode="create", memory_area=True)
    lines = _fm_lines(out)
    # Exactly one server-asserted line per key; the spoofed values are gone.
    assert lines.count("author: 'agent:agent-a'") == 1
    assert lines.count("written_via: mcp") == 1
    assert lines.count("memory_status: unconsolidated") == 1
    assert "someone-else" not in out
    assert "telepathy" not in out
    assert "consolidated\n" not in out.replace("unconsolidated", "")


@pytest.mark.parametrize("mode", ["replace", "append"])
def test_replace_and_append_carry_author_and_status_from_prior(mode) -> None:
    prior = (
        "---\n"
        "title: x\n"
        "author: 'agent:creator'\n"
        "memory_status: unconsolidated\n"
        "written_via: mcp\n"
        "---\n" + _BODY
    )
    # The CLIENT content re-sends the note (a well-behaved client echoes it).
    out = stamp_provenance(prior, agent="editor", mode=mode, memory_area=True, prior=prior)
    lines = _fm_lines(out)
    # Last-writer attribution asserted; the create-time author and the
    # consolidation state are carried forward FROM PRIOR (not the client body).
    assert "last_written_by: 'agent:editor'" in lines
    assert "author: 'agent:creator'" in lines
    assert "memory_status: unconsolidated" in lines
    assert lines.count("written_via: mcp") == 1
    assert _body_of(out) == _BODY


@pytest.mark.parametrize("mode", ["replace", "append"])
def test_replace_and_append_reject_client_spoofed_provenance(mode) -> None:
    # F013: the create-time author + consolidation state come from the
    # PRIOR note; the client's forged values in the new body are ignored.
    prior = (
        "---\n"
        "title: x\n"
        "author: 'agent:creator'\n"
        "memory_status: unconsolidated\n"
        "written_via: mcp\n"
        "---\n" + _BODY
    )
    forged = (
        "---\n"
        "title: x\n"
        "author: 'agent:victim'\n"           # spoof create-time author
        "memory_status: consolidated\n"      # self-promote past the gate
        "last_written_by: 'agent:someone'\n"
        "written_via: telepathy\n"
        "---\n" + _BODY
    )
    out = stamp_provenance(forged, agent="editor", mode=mode, memory_area=True, prior=prior)
    lines = _fm_lines(out)
    assert "last_written_by: 'agent:editor'" in lines
    assert "author: 'agent:creator'" in lines            # from prior, not 'victim'
    assert "memory_status: unconsolidated" in lines       # NOT self-promoted
    assert "victim" not in out
    assert "telepathy" not in out
    assert "someone" not in out
    assert "consolidated\n" not in out.replace("unconsolidated", "")
    assert lines.count("written_via: mcp") == 1


def test_replace_strips_whitespace_padded_spoof_key() -> None:
    # 'author : x' (space before the colon) is still a YAML key; the strip
    # must be whitespace-tolerant or a duplicate spoofed line survives.
    prior = "---\ntitle: x\nauthor: 'agent:creator'\n---\n" + _BODY
    forged = "---\ntitle: x\nauthor : 'agent:victim'\n---\n" + _BODY
    out = stamp_provenance(forged, agent="editor", mode="replace", memory_area=False, prior=prior)
    assert "victim" not in out
    assert "author: 'agent:creator'" in _fm_lines(out)


def test_replace_without_frontmatter_prepends_block() -> None:
    out = stamp_provenance(_BODY, agent="editor", mode="replace", memory_area=False)
    assert _fm_lines(out) == ["last_written_by: 'agent:editor'", "written_via: mcp"]


def test_unterminated_fence_is_not_frontmatter() -> None:
    # notes._split_frontmatter refuses an unterminated fence; stamping
    # must agree and prepend a fresh block instead of splicing into it.
    weird = "---\ntitle: never closed\nbody line\n"
    out = stamp_provenance(weird, agent="agent-a", mode="create", memory_area=False)
    assert out.endswith(weird)  # original content intact as body
    assert _fm_lines(out)[0] == "author: 'agent:agent-a'"


# ------------------------------------------------- frontmatter_signature

_REL_NOTE = (
    "---\n"
    "title: anna\n"
    "topics: [People]\n"
    "relations:\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "---\n"
    "Body.\n"
)


def test_signature_unchanged_by_body_only_edit() -> None:
    edited = _REL_NOTE.replace("Body.\n", "A completely different body.\n")
    assert frontmatter_signature(_REL_NOTE) == frontmatter_signature(edited)


def test_signature_changes_on_topics_edit() -> None:
    edited = _REL_NOTE.replace("topics: [People]", "topics: [People, Hiring]")
    assert frontmatter_signature(_REL_NOTE) != frontmatter_signature(edited)


def test_signature_changes_on_relations_edit() -> None:
    edited = _REL_NOTE.replace("organisations/acme", "organisations/initech")
    assert frontmatter_signature(_REL_NOTE) != frontmatter_signature(edited)


def test_signature_normalises_topic_variants() -> None:
    # Case/punctuation variants slugify identically (how concepts group),
    # so they must not count as a graph change.
    a = _REL_NOTE.replace("topics: [People]", "topics: [Behaviour-Driven Development]")
    b = _REL_NOTE.replace("topics: [People]", "topics: [behaviour driven development]")
    assert frontmatter_signature(a) == frontmatter_signature(b)


def test_signature_of_unstamped_content_is_empty() -> None:
    assert frontmatter_signature("# just a body\n") == ((), ())


# --------------------------------------------------------- end to end

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


def _audit_rows(root: Path) -> list[dict]:
    path = root / "logs" / "mcp-audit.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


def test_write_tools_stamp_commit_and_audit(tmp_path: Path) -> None:
    root = _make_vault(tmp_path)
    cfg = _cfg(root)
    runtime = _runtime(root)
    token = AGENT_VAR.set("agent-a")
    try:
        # -- create outside the memory area: author + written_via, no status.
        res = tool_create_note(
            cfg, runtime,
            path="knowledge/notes/anna.md",
            content="---\ntitle: anna\nauthor: 'agent:spoof'\n---\nHello.\n",
        )
        on_disk = (root / "knowledge/notes/anna.md").read_text(encoding="utf-8")
        assert "author: 'agent:agent-a'" in on_disk
        assert "written_via: mcp" in on_disk
        assert "spoof" not in on_disk
        assert "memory_status" not in on_disk
        assert on_disk.endswith("Hello.\n")
        assert _git(root, "log", "-1", "--format=%s") == \
            "mcp(agent-a): create note knowledge/notes/anna.md"
        assert res.committed and res.commit_sha
        assert res.pushed is False           # pushes are async now
        assert res.push_state == "disabled"  # worker is off in this Runtime
        assert res.index_refresh == "off"    # refresher is off too

        # -- create in the memory area: memory_status appears.
        tool_create_note(
            cfg, runtime,
            path="knowledge/assistant/inbox/fact.md",
            content="A fresh observation.\n",
        )
        mem = (root / "knowledge/assistant/inbox/fact.md").read_text(encoding="utf-8")
        assert "memory_status: unconsolidated" in mem
        assert "author: 'agent:agent-a'" in mem

        # -- replace: last_written_by stamped, create-time author survives.
        res2 = tool_replace_note(
            cfg, runtime,
            path="knowledge/notes/anna.md",
            content=on_disk.replace("Hello.", "Hello again."),
        )
        replaced = (root / "knowledge/notes/anna.md").read_text(encoding="utf-8")
        assert "author: 'agent:agent-a'" in replaced
        assert "last_written_by: 'agent:agent-a'" in replaced
        assert replaced.endswith("Hello again.\n")
        assert _git(root, "log", "-1", "--format=%s") == \
            "mcp(agent-a): replace note knowledge/notes/anna.md"
        assert res2.push_state == "disabled" and res2.index_refresh == "off"

        # -- a refusal is audited too.
        with pytest.raises(ToolError):
            tool_create_note(cfg, runtime, path="knowledge/notes/anna.md", content="x")
    finally:
        AGENT_VAR.reset(token)

    rows = _audit_rows(root)
    assert [r["outcome"].split(":")[0] for r in rows] == ["ok", "ok", "ok", "refused"]
    assert all(r["agent"] == "agent-a" for r in rows)
    assert rows[0]["tool"] == "vault_create_note"
    assert rows[0]["path"] == "knowledge/notes/anna.md"
    assert rows[0]["detail"].startswith("commit=")
    assert "already exists" in rows[3]["outcome"]
