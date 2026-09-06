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
from typing import TYPE_CHECKING, Literal, TypedDict

import pytest
import yaml

# Type-only import of the shared harness contracts. At runtime the module
# is pytest's ``conftest``; ``tests.conftest`` is the name mypy resolves
# (explicit_package_bases), and importing it for real would collide with an
# unrelated installed ``tests`` package. Annotations are postponed, so
# nothing here is evaluated at import time.
if TYPE_CHECKING:
    from tests.conftest import GitRunner, McpEnv

from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.errors import ToolError
from mcp_server.identity import AGENT_VAR
from mcp_server.provenance import ProvenanceError, frontmatter_signature, stamp_provenance
from mcp_server import tools as tools_mod
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import tool_append_to_note, tool_create_note, tool_replace_note


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
def test_replace_and_append_carry_author_and_status_from_prior(
    mode: Literal["replace", "append"],
) -> None:
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
def test_replace_and_append_reject_client_spoofed_provenance(
    mode: Literal["replace", "append"],
) -> None:
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


@pytest.mark.parametrize("qkey", ['"author"', "'author'", '"author" ', "'memory_status'"])
def test_create_strips_quoted_spoof_key(qkey: str) -> None:
    # A QUOTED YAML key ("author": / 'author':) parses as the same key but
    # slips past an unquoted-only strip regex, leaving a forged line on disk.
    forged = (
        "---\n"
        "title: x\n"
        f"{qkey}: 'agent:victim'\n"
        "---\n" + _BODY
    )
    out = stamp_provenance(forged, agent="agent-a", mode="create", memory_area=True)
    assert "victim" not in out
    # The server's own line is the only author/memory_status line present.
    assert "author: 'agent:agent-a'" in _fm_lines(out)


def test_replace_quoted_spoof_key_without_prior_key_is_not_forgeable() -> None:
    # The dangerous path: the prior note lacks a server author line (a
    # hand-authored / pre-provenance Obsidian note), so nothing is carried
    # forward. A quoted spoof key must NOT become the effective author.
    import yaml

    prior = "---\ntitle: x\n---\n" + _BODY  # no author line at all
    forged = (
        "---\n"
        "title: x\n"
        '"author": \'agent:victim\'\n'
        '"memory_status": consolidated\n'
        "---\n" + _BODY
    )
    out = stamp_provenance(forged, agent="editor", mode="replace", memory_area=True, prior=prior)
    assert "victim" not in out
    # Parse the real effective values, not just substring presence.
    parsed = yaml.safe_load(out.split("\n---\n", 1)[0][4:])
    assert parsed.get("author") != "agent:victim"
    assert parsed.get("memory_status") != "consolidated"


@pytest.mark.parametrize("mode", ["replace", "append"])
def test_complex_key_author_spoof_without_prior_key_is_refused(
    mode: Literal["replace", "append"],
) -> None:
    # F1: a YAML complex key (`? author` / `: agent:evil`) resolves the
    # same mapping key as `author: agent:evil` but never matches the
    # line-strip regex, which only recognises `<key>:` at line start. The
    # prior note has no server author line to carry forward, so nothing
    # is asserted for `author` this call — the forged value must not
    # survive into the parsed result.
    prior = "---\ntitle: x\n---\n" + _BODY
    forged = (
        "---\n"
        "title: y\n"
        "? author\n"
        ": agent:evil\n"
        "---\n" + _BODY
    )
    with pytest.raises(ToolError, match="author"):
        stamp_provenance(forged, agent="editor", mode=mode, memory_area=True, prior=prior)


@pytest.mark.parametrize("mode", ["replace", "append"])
def test_merge_key_author_and_status_spoof_is_refused(
    mode: Literal["replace", "append"],
) -> None:
    # F1: a YAML merge key/anchor (`z: &a {author: .., memory_status: ..}`
    # + `<<: *a`) resolves both `author` and `memory_status` without ever
    # writing a line matching `^\s*["']?author["']?\s*:`. `memory_status:
    # consolidated` here is a self-promotion past the consolidation gate
    # in a memory area — must be refused just as hard as the author spoof.
    prior = "---\ntitle: x\n---\n" + _BODY
    forged = (
        "---\n"
        "title: y\n"
        "z: &a {author: agent:evil, memory_status: consolidated}\n"
        "<<: *a\n"
        "---\n" + _BODY
    )
    with pytest.raises(ToolError, match=r"survives stamping"):
        stamp_provenance(forged, agent="editor", mode=mode, memory_area=True, prior=prior)


def test_replace_without_frontmatter_prepends_block() -> None:
    out = stamp_provenance(_BODY, agent="editor", mode="replace", memory_area=False)
    assert _fm_lines(out) == ["last_written_by: 'agent:editor'", "written_via: mcp"]


def test_unterminated_fence_raises() -> None:
    # notes._split_frontmatter refuses an unterminated fence, same as any
    # other fenced-but-unparseable content (AUD-001): a leading '---' is a
    # commitment to a real frontmatter block, so stamping must refuse
    # rather than quietly prepend a second fence on top of it.
    weird = "---\ntitle: never closed\nbody line\n"
    with pytest.raises(ProvenanceError, match="no closing"):
        stamp_provenance(weird, agent="agent-a", mode="create", memory_area=False)


def test_no_fence_at_all_still_prepends_minimal_block() -> None:
    # A body that merely CONTAINS a '---' later on (not as its first line)
    # is not a frontmatter fence at all, so the old prepend fallback still
    # applies — only a LEADING fence changes AUD-001's behaviour.
    body = "# Title\n\nSome text.\n\n---\n\nA horizontal rule, not frontmatter.\n"
    out = stamp_provenance(body, agent="agent-a", mode="create", memory_area=False)
    assert _fm_lines(out) == ["author: 'agent:agent-a'", "written_via: mcp"]
    assert out == "---\nauthor: 'agent:agent-a'\nwritten_via: mcp\n---\n" + body


def test_fenced_yaml_error_raises_provenance_error() -> None:
    # AUD-001: an unquoted colon inside a plain scalar is invalid YAML
    # ("mapping values are not allowed here"); _split_frontmatter can't
    # parse it, so stamping must refuse rather than bury it under a second
    # server-authored fence.
    content = (
        "---\n"
        'title: fps — "Quarterly Review: a fever dream"\n'
        "---\n" + _BODY
    )
    with pytest.raises(ProvenanceError, match="mapping values are not allowed here"):
        stamp_provenance(content, agent="agent-a", mode="create", memory_area=False)


def test_fenced_yaml_list_raises_provenance_error() -> None:
    # AUD-001: valid YAML that isn't a mapping (a bare list document) is
    # just as unparseable to every downstream frontmatter reader.
    content = "---\n- a\n- b\n---\n" + _BODY
    with pytest.raises(ProvenanceError, match="not a mapping"):
        stamp_provenance(content, agent="agent-a", mode="create", memory_area=False)


def test_indented_block_mapping_raises_provenance_error() -> None:
    # AUD-002: '  title: x' is valid (indented) YAML on its own, so the
    # initial parse succeeds — but splicing unindented server keys in
    # just before the closing fence produces a document mixing an
    # indented block with unindented top-level keys, which no longer
    # parses at all. The post-condition must catch this rather than
    # commit unparseable frontmatter.
    content = "---\n  title: x\n---\nbody\n"
    with pytest.raises(ProvenanceError, match="does not parse"):
        stamp_provenance(content, agent="agent-a", mode="create", memory_area=False)


def test_flow_mapping_raises_provenance_error() -> None:
    # AUD-002: a flow-mapping document ('{title: x}') parses fine alone,
    # but appending block-style server keys after it is the same collision
    # as the indented case.
    content = "---\n{title: x}\n---\nbody\n"
    with pytest.raises(ProvenanceError, match="does not parse"):
        stamp_provenance(content, agent="agent-a", mode="create", memory_area=False)


def test_ordinary_frontmatter_still_stamps_and_parses() -> None:
    # Sanity check that AUD-001/AUD-002 add no false positives: well-formed
    # frontmatter still stamps normally and the result round-trips through
    # yaml.safe_load with the asserted keys present.
    import yaml

    out = stamp_provenance(_WITH_FM, agent="agent-a", mode="create", memory_area=False)
    parsed = yaml.safe_load(out.split("\n---\n", 1)[0][4:])
    assert parsed["author"] == "agent:agent-a"
    assert parsed["written_via"] == "mcp"
    assert parsed["title"] == "Anna Kowalska"


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


class _AuditRow(TypedDict):
    ts: str
    agent: str
    tool: str
    path: str | None
    outcome: str
    detail: str | None


def _audit_rows(root: Path) -> list[_AuditRow]:
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
        with pytest.raises(ToolError, match=r"file already exists"):
            tool_create_note(cfg, runtime, path="knowledge/notes/anna.md", content="x")
    finally:
        AGENT_VAR.reset(token)

    rows = _audit_rows(root)
    assert [r["outcome"].split(":")[0] for r in rows] == ["ok", "ok", "ok", "refused"]
    assert all(r["agent"] == "agent-a" for r in rows)
    assert rows[0]["tool"] == "vault_create_note"
    assert rows[0]["path"] == "knowledge/notes/anna.md"
    detail = rows[0]["detail"]
    assert detail is not None and detail.startswith("commit=")
    assert "already exists" in rows[3]["outcome"]


def test_create_tool_refuses_unparseable_frontmatter_as_a_refusal(tmp_path: Path) -> None:
    # AUD-001 through the real tool path: _audited_write must catch
    # ProvenanceError (a ToolError subclass) and record it as a refusal,
    # not commit a note with client frontmatter buried under a second,
    # server-authored fence.
    root = _make_vault(tmp_path)
    cfg = _cfg(root)
    runtime = _runtime(root)
    token = AGENT_VAR.set("agent-a")
    try:
        with pytest.raises(ToolError, match="mapping values are not allowed here"):
            tool_create_note(
                cfg, runtime,
                path="knowledge/notes/fps.md",
                content=(
                    "---\n"
                    'title: fps — "Quarterly Review: a fever dream"\n'
                    "---\nBody.\n"
                ),
            )
    finally:
        AGENT_VAR.reset(token)

    assert not (root / "knowledge/notes/fps.md").exists()
    rows = _audit_rows(root)
    assert len(rows) == 1
    assert rows[0]["outcome"].startswith("refused")


# ---------------- AUD-067: the YAML-forgery shapes through the write tools
# The unit tests above pin stamp_provenance itself. These drive the real
# create/replace/append tools over a throwaway git vault (the shared
# mcp_vault/make_cfg/make_runtime harness, via ``mcp_env``), because the
# outcome is NOT uniform across the three verbs: only some combinations
# refuse, and the ones that don't are neutralised rather than rejected.

_SIMPLE_BODY = "Body.\n"

# `? author` / `: agent:evil` resolves the same mapping key as
# `author: agent:evil` but matches no `^<key>:` line, so the strip loop
# never sees it.
_COMPLEX_KEY = "---\ntitle: y\n? author\n: agent:evil\n---\n" + _SIMPLE_BODY

# A merge key resolves BOTH author and memory_status from an anchor,
# again without writing a line the strip loop can match.
_MERGE_KEY = (
    "---\n"
    "title: y\n"
    "z: &a {author: agent:evil, memory_status: consolidated}\n"
    "<<: *a\n"
    "---\n" + _SIMPLE_BODY
)

_PRIOR_UNSTAMPED = "---\ntitle: x\n---\n" + _SIMPLE_BODY
_PRIOR_STAMPED = (
    "---\ntitle: x\nauthor: 'agent:creator'\nwritten_via: mcp\n---\n" + _SIMPLE_BODY
)
_PRIOR_STAMPED_MEMORY = (
    "---\n"
    "title: x\n"
    "author: 'agent:creator'\n"
    "memory_status: unconsolidated\n"
    "written_via: mcp\n"
    "---\n" + _SIMPLE_BODY
)

_NOTE = "knowledge/notes/x.md"
_MEMORY_NOTE = "knowledge/assistant/inbox/x.md"


def _parsed_fm(text: str) -> dict[str, object]:
    """The frontmatter mapping a YAML reader actually resolves. Where the
    server neutralises a forgery rather than refusing it, the forged text
    is still on disk — only the PARSED value says whether it took."""
    loaded = yaml.safe_load(text[4:].split("\n---\n", 1)[0])
    assert isinstance(loaded, dict)
    return loaded


def _write_seed(root: Path, rel: str, text: str) -> None:
    """Put a note on disk directly — the shapes below can only reach a
    note's frontmatter through something that is not an MCP write tool."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_note(root: Path, run_git: GitRunner, rel: str, text: str) -> str:
    """Same, then commit it. Returns HEAD, so a refusal can be pinned to
    'no new commit' as well as 'no new bytes'."""
    _write_seed(root, rel, text)
    run_git(root, "add", "-A")
    run_git(root, "commit", "-qm", "seed")
    return run_git(root, "rev-parse", "HEAD")


# -- create: the complex key is neutralised, not refused ---------------------

def test_create_neutralises_complex_key_author_spoof(mcp_env: McpEnv) -> None:
    # No refusal here: the server's own `author:` line is inserted LAST, and
    # for a duplicate mapping key every YAML reader takes the last one — so
    # the post-condition sees the server's value and passes.
    root, cfg, runtime = mcp_env
    tool_create_note(cfg, runtime, path=_NOTE, content=_COMPLEX_KEY)
    on_disk = (root / _NOTE).read_text(encoding="utf-8")
    # The forged construct is still on disk verbatim — neutralised, not stripped.
    assert "? author" in on_disk and "agent:evil" in on_disk
    parsed = _parsed_fm(on_disk)
    assert parsed["author"] == "agent:agent-a"
    assert parsed["written_via"] == "mcp"
    assert _audit_rows(root)[-1]["outcome"] == "ok"


def test_create_in_memory_area_neutralises_merge_key_spoof(mcp_env: McpEnv) -> None:
    # In a memory area the server asserts memory_status too, so both forged
    # keys are overridden by later explicit keys and the write goes through.
    root, cfg, runtime = mcp_env
    tool_create_note(cfg, runtime, path=_MEMORY_NOTE, content=_MERGE_KEY)
    parsed = _parsed_fm((root / _MEMORY_NOTE).read_text(encoding="utf-8"))
    assert parsed["author"] == "agent:agent-a"
    assert parsed["memory_status"] == "unconsolidated"
    # The evil values survive only under `z`, which is not a provenance key.
    assert parsed["z"] == {"author": "agent:evil", "memory_status": "consolidated"}


def test_create_outside_memory_area_refuses_merge_key_status_spoof(
    mcp_env: McpEnv, run_git: GitRunner
) -> None:
    # Outside a memory area the server asserts NO memory_status, so the
    # merged `consolidated` is the effective value and nothing overrides it.
    root, cfg, runtime = mcp_env
    head = _seed_note(root, run_git, "knowledge/notes/seed.md", "seed\n")
    with pytest.raises(ToolError, match=r"memory_status survives stamping"):
        tool_create_note(cfg, runtime, path=_NOTE, content=_MERGE_KEY)
    assert not (root / _NOTE).exists()
    assert run_git(root, "rev-parse", "HEAD") == head
    assert _audit_rows(root)[-1]["outcome"].startswith("refused")


# -- replace: refuses only when the prior note has no server author ----------

@pytest.mark.parametrize("forged", [_COMPLEX_KEY, _MERGE_KEY], ids=["complex", "merge"])
def test_replace_refuses_forged_author_when_prior_has_none(
    mcp_env: McpEnv, run_git: GitRunner, forged: str
) -> None:
    # The dangerous case: a hand-authored / pre-provenance note has no
    # server `author:` line to carry forward, so the server asserts none
    # this call and the forged value would BE the effective author.
    root, cfg, runtime = mcp_env
    head = _seed_note(root, run_git, _NOTE, _PRIOR_UNSTAMPED)
    with pytest.raises(ToolError, match=r"author survives stamping"):
        tool_replace_note(cfg, runtime, path=_NOTE, content=forged)
    assert (root / _NOTE).read_text(encoding="utf-8") == _PRIOR_UNSTAMPED
    assert run_git(root, "rev-parse", "HEAD") == head
    assert _audit_rows(root)[-1]["outcome"].startswith("refused")


def test_replace_carries_prior_author_over_a_complex_key_spoof(mcp_env: McpEnv) -> None:
    # With an author line to carry forward there is no refusal: the carried
    # server line lands after the forgery and wins the duplicate key.
    root, cfg, runtime = mcp_env
    _write_seed(root, _NOTE, _PRIOR_STAMPED)
    tool_replace_note(cfg, runtime, path=_NOTE, content=_COMPLEX_KEY)
    parsed = _parsed_fm((root / _NOTE).read_text(encoding="utf-8"))
    assert parsed["author"] == "agent:creator"          # not agent:evil
    assert parsed["last_written_by"] == "agent:agent-a"


def test_replace_refuses_merge_key_status_spoof_outside_a_memory_area(
    mcp_env: McpEnv, run_git: GitRunner
) -> None:
    # Author is carried forward and wins, but memory_status is asserted by
    # nobody outside a memory area — so the merged `consolidated` survives
    # and the write is refused on THAT key, not on author.
    root, cfg, runtime = mcp_env
    head = _seed_note(root, run_git, _NOTE, _PRIOR_STAMPED)
    with pytest.raises(ToolError, match=r"memory_status survives stamping"):
        tool_replace_note(cfg, runtime, path=_NOTE, content=_MERGE_KEY)
    assert (root / _NOTE).read_text(encoding="utf-8") == _PRIOR_STAMPED
    assert run_git(root, "rev-parse", "HEAD") == head
    assert _audit_rows(root)[-1]["outcome"].startswith("refused")


def test_replace_in_memory_area_neutralises_merge_key_spoof(mcp_env: McpEnv) -> None:
    # Both keys carried forward from the prior note, both re-asserted last.
    root, cfg, runtime = mcp_env
    _write_seed(root, _MEMORY_NOTE, _PRIOR_STAMPED_MEMORY)
    tool_replace_note(cfg, runtime, path=_MEMORY_NOTE, content=_MERGE_KEY)
    parsed = _parsed_fm((root / _MEMORY_NOTE).read_text(encoding="utf-8"))
    assert parsed["author"] == "agent:creator"
    assert parsed["memory_status"] == "unconsolidated"   # NOT self-promoted


# -- append: the client's fence is body text, so forging it is a no-op -------

@pytest.mark.parametrize("forged", [_COMPLEX_KEY, _MERGE_KEY], ids=["complex", "merge"])
def test_append_of_a_forged_fence_is_inert_body_text(
    mcp_env: McpEnv, forged: str
) -> None:
    # Append stamps the COMBINED text, whose frontmatter is the one already
    # on disk — the client's fence lands below it, in the body, where it is
    # not frontmatter at all. Nothing to refuse and nothing forged.
    root, cfg, runtime = mcp_env
    _write_seed(root, _NOTE, _PRIOR_STAMPED)
    tool_append_to_note(cfg, runtime, path=_NOTE, content=forged)
    on_disk = (root / _NOTE).read_text(encoding="utf-8")
    assert on_disk.endswith(forged)                      # verbatim, in the body
    parsed = _parsed_fm(on_disk)
    assert parsed["author"] == "agent:creator"
    assert parsed["last_written_by"] == "agent:agent-a"
    assert "memory_status" not in parsed
    assert _audit_rows(root)[-1]["outcome"] == "ok"


@pytest.mark.parametrize("forged", [_COMPLEX_KEY, _MERGE_KEY], ids=["complex", "merge"])
def test_append_refuses_when_the_forged_shape_is_already_on_disk(
    mcp_env: McpEnv, run_git: GitRunner, forged: str
) -> None:
    # The shape only reaches the frontmatter if it is ALREADY the note's
    # own frontmatter (dropped in over Obsidian/git, not via the tools).
    # Then the append inherits it, carries no server author forward, and
    # the post-condition catches the forgery before anything is written.
    root, cfg, runtime = mcp_env
    head = _seed_note(root, run_git, _NOTE, forged)
    with pytest.raises(ToolError, match=r"author survives stamping"):
        tool_append_to_note(cfg, runtime, path=_NOTE, content="more text\n")
    assert (root / _NOTE).read_text(encoding="utf-8") == forged
    assert run_git(root, "rev-parse", "HEAD") == head
    assert _audit_rows(root)[-1]["outcome"].startswith("refused")
