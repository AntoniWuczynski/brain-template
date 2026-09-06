"""Happy-path coverage for the MCP read tools that the existing suite only
ever exercises on their refusal paths (F2), plus the two write/read tools
that were never executed at all (F3): ``tool_search``, ``tool_related``,
``tool_list``, ``tool_metadata_query``, ``tool_drop_inbox_file``.

Same throwaway-vault + ServerConfig + Runtime harness as
tests/test_mcp_happy_path.py; the real embedding model is never loaded —
``tool_search`` is exercised in ``mode="lexical"`` (no model load, per
``ingest_lib.semantic.search``'s docstring) over a hand-seeded index, the
same fixture approach tests/test_reindex.py and tests/test_mcp_chunk_context.py
use.
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ingest_lib.config import paths_for_root
from ingest_lib.connections import rebuild_connections
from ingest_lib.metadata import IndexRecord, append_record

from mcp_server import tools as tools_mod
from mcp_server import tools_read as tools_read_mod
from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.errors import ToolError
from mcp_server.identity import AGENT_VAR
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import tool_drop_inbox_file
from mcp_server.tools_read import (
    tool_list,
    tool_metadata_query,
    tool_related,
    tool_search,
)

_LOG = logging.getLogger("test")


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


def _cfg(root: Path) -> ServerConfig:
    return ServerConfig(
        vault_root=root,
        tokens=(("x" * 24, "agent-a"),),
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


@pytest.fixture(autouse=True)
def _roomy_rate_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Search/read/write buckets are module-global; give every test a fresh
    # roomy one so order across the whole pytest process never starves it.
    monkeypatch.setattr(tools_mod, "_write_bucket", tools_mod._RateBucket(10_000))
    monkeypatch.setattr(tools_read_mod, "_search_bucket", tools_mod._RateBucket(10_000))
    monkeypatch.setattr(tools_read_mod, "_read_bucket", tools_mod._RateBucket(10_000))


_Env = tuple[Path, ServerConfig, Runtime]


@pytest.fixture()
def env(tmp_path: Path) -> Iterator[_Env]:
    root = _make_vault(tmp_path)
    token = AGENT_VAR.set("agent-a")
    try:
        yield root, _cfg(root), _runtime(root)
    finally:
        AGENT_VAR.reset(token)


# ------------------------------------------------------------ tool_search

def _seed_search_index(root: Path, rows: list[dict[str, str | int]]) -> None:
    meta = root / "metadata" / "embeddings_meta.jsonl"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # mode="lexical" never touches the vectors, but search() requires the
    # row count to match the meta line count before it will run at all.
    np.save(root / "metadata" / "embeddings.npy", np.zeros((len(rows), 4), dtype=np.float32))


def _write_note(root: Path, rel: str) -> None:
    note = root / rel
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("body\n", encoding="utf-8")


def test_tool_search_returns_hits_gated_by_read_policy(env: _Env) -> None:
    root, cfg, runtime = env
    _write_note(root, "knowledge/notes/tcp.md")
    _seed_search_index(root, [
        {
            "source_relative_path": "knowledge/notes/tcp.md", "source_hash": "h1",
            "title": "TCP congestion control", "chunk_idx": 0,
            "text": "slow start and congestion avoidance in TCP",
            "origin": "knowledge-note",
        },
        {
            # A hit whose backing artifact the read policy denies (a
            # DENY_READ_NAME, gated as-is since origin marks it a
            # knowledge-note): must be dropped, not returned or crash the call.
            "source_relative_path": ".env", "source_hash": "h2",
            "title": "leaked secret", "chunk_idx": 0,
            "text": "congestion secret token",
            "origin": "knowledge-note",
        },
    ])

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)

    assert len(out.hits) == 1
    assert out.hits[0].source_relative_path == "knowledge/notes/tcp.md"
    assert out.hits[0].title == "TCP congestion control"

    access_path = root / "logs" / "mcp-access.jsonl"
    rows = [json.loads(ln) for ln in access_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["tool"] == "vault_search"
    assert rows[0]["paths"] == ["knowledge/notes/tcp.md"]


def test_tool_search_rejects_bad_top_k_and_mode(env: _Env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="top_k"):
        tool_search(cfg, runtime, query="x", top_k=0)
    with pytest.raises(ToolError, match="mode"):
        tool_search(cfg, runtime, query="x", mode="bogus")


def _write_entity_note(
    root: Path, rel: str, *, title: str, aliases: list[str] | None = None
) -> None:
    note = root / rel
    note.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"title: {title}"]
    if aliases:
        lines.append("aliases:")
        lines.extend(f"  - {a}" for a in aliases)
    lines += ["---", "body", ""]
    note.write_text("\n".join(lines), encoding="utf-8")


def test_tool_search_evidence_hints_exists_probable_and_unknown(env: _Env) -> None:
    root, cfg, runtime = env
    _write_entity_note(
        root, "knowledge/people/anna-kowalska.md",
        title="Anna Kowalska", aliases=["Anna K"],
    )
    _write_note(root, "knowledge/notes/note-a.md")
    _write_note(root, "knowledge/notes/note-b.md")
    _write_note(root, "knowledge/notes/note-c.md")
    _write_note(root, "knowledge/meetings/2026/2026-06-12-standup.md")
    _seed_search_index(root, [
        # A hit that is itself an entity note under a hint dir.
        {
            "source_relative_path": "knowledge/meetings/2026/2026-06-12-standup.md",
            "source_hash": "h0", "title": "Standup", "chunk_idx": 0,
            "text": "congestion agenda item", "origin": "knowledge-note",
        },
        # A hit whose title matches an entity's alias exactly, but the hit
        # itself lives outside the entity dirs.
        {
            "source_relative_path": "knowledge/notes/note-a.md", "source_hash": "h1",
            "title": "Anna K", "chunk_idx": 0,
            "text": "congestion meeting notes", "origin": "knowledge-note",
        },
        # A near-miss on the entity's title (one deletion) — probable, not
        # exact (must differ from the title/alias to avoid the exists path).
        {
            "source_relative_path": "knowledge/notes/note-b.md", "source_hash": "h2",
            "title": "Ana Kowalska", "chunk_idx": 0,
            "text": "congestion follow-up", "origin": "knowledge-note",
        },
        # No relation to any entity at all.
        {
            "source_relative_path": "knowledge/notes/note-c.md", "source_hash": "h3",
            "title": "Unrelated Topic", "chunk_idx": 0,
            "text": "congestion unrelated", "origin": "knowledge-note",
        },
    ])

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)
    by_path = {h.source_relative_path: h for h in out.hits}

    meeting = by_path["knowledge/meetings/2026/2026-06-12-standup.md"]
    assert meeting.evidence.status == "exists"
    assert meeting.evidence.node_id == "meetings/2026/2026-06-12-standup"

    alias_hit = by_path["knowledge/notes/note-a.md"]
    assert alias_hit.evidence.status == "exists"
    assert alias_hit.evidence.node_id == "people/anna-kowalska"

    near_hit = by_path["knowledge/notes/note-b.md"]
    assert near_hit.evidence.status == "probable"
    assert near_hit.evidence.node_id == "people/anna-kowalska"

    unknown_hit = by_path["knowledge/notes/note-c.md"]
    assert unknown_hit.evidence.status == "unknown"
    assert unknown_hit.evidence.node_id is None


def test_tool_search_evidence_hint_maps_project_log_entry_to_project(env: _Env) -> None:
    # F2: a note under a project folder that is not the overview itself
    # (a dated ``log/`` entry) is not a node of its own — it belongs to the
    # project's overview note.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/claude/claude.md", title="Claude")
    _write_entity_note(
        root, "knowledge/projects/claude/log/2026-07-20.md", title="2026-07-20",
    )
    _seed_search_index(root, [{
        "source_relative_path": "knowledge/projects/claude/log/2026-07-20.md",
        "source_hash": "h0", "title": "2026-07-20", "chunk_idx": 0,
        "text": "congestion log entry", "origin": "knowledge-note",
    }])

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)
    hit = out.hits[0]
    assert hit.evidence.status == "exists"
    assert hit.evidence.node_id == "projects/claude/claude"


def test_tool_search_evidence_hint_excludes_non_entity_notes(env: _Env) -> None:
    # F5: the assistant PROFILE and an unapproved inbox proposal are not
    # entity notes — neither the path branch (a hit ON them) nor a title
    # match (a hit whose title happens to equal one of theirs) may report
    # them as an existing graph node.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/assistant/PROFILE.md", title="Assistant Profile")
    _write_entity_note(
        root, "knowledge/assistant/inbox/2026-09-01-server-ssh.md",
        title="Home server online with SSH access",
    )
    _write_note(root, "knowledge/notes/mentions.md")
    _seed_search_index(root, [
        {
            "source_relative_path": "knowledge/assistant/PROFILE.md",
            "source_hash": "h0", "title": "Assistant Profile", "chunk_idx": 0,
            "text": "congestion profile", "origin": "knowledge-note",
        },
        {
            "source_relative_path": "knowledge/assistant/inbox/2026-09-01-server-ssh.md",
            "source_hash": "h1", "title": "Home server online with SSH access",
            "chunk_idx": 0, "text": "congestion inbox", "origin": "knowledge-note",
        },
        {
            "source_relative_path": "knowledge/notes/mentions.md",
            "source_hash": "h2", "title": "Assistant Profile", "chunk_idx": 0,
            "text": "congestion mention", "origin": "knowledge-note",
        },
    ])

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)
    by_path = {h.source_relative_path: h for h in out.hits}

    assert by_path["knowledge/assistant/PROFILE.md"].evidence.status == "unknown"
    assert (
        by_path["knowledge/assistant/inbox/2026-09-01-server-ssh.md"].evidence.status
        == "unknown"
    )
    assert by_path["knowledge/notes/mentions.md"].evidence.status == "unknown"


def test_tool_search_evidence_hint_superseded_note_resolves_to_survivor(env: _Env) -> None:
    # F1 + F10: a hit on a superseded note (the path branch) and a hit whose
    # title merely matches the superseded note's title (the exact-key
    # branch) must both resolve through live_entity_id to the survivor —
    # the dead node is never handed back as the entity to link.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/people/anna-kowalska.md", title="Anna Kowalska")
    dead = root / "knowledge/people/annakowalskaexamplecom.md"
    dead.parent.mkdir(parents=True, exist_ok=True)
    dead.write_text(
        "---\ntitle: anna.kowalska@example.com\n"
        "superseded_by: people/anna-kowalska\n---\nbody\n",
        encoding="utf-8",
    )
    _write_note(root, "knowledge/notes/mentions.md")
    _seed_search_index(root, [
        {
            "source_relative_path": "knowledge/people/annakowalskaexamplecom.md",
            "source_hash": "h0", "title": "anna.kowalska@example.com", "chunk_idx": 0,
            "text": "congestion note", "origin": "knowledge-note",
        },
        {
            "source_relative_path": "knowledge/notes/mentions.md",
            "source_hash": "h1", "title": "anna.kowalska@example.com", "chunk_idx": 0,
            "text": "congestion mention", "origin": "knowledge-note",
        },
    ])

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)
    by_path = {h.source_relative_path: h for h in out.hits}

    for path in (
        "knowledge/people/annakowalskaexamplecom.md",
        "knowledge/notes/mentions.md",
    ):
        hit = by_path[path]
        assert hit.evidence.status == "exists"
        assert hit.evidence.node_id == "people/anna-kowalska"


def test_tool_search_skips_entity_index_build_when_no_hits(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F8: an empty hit list must not pay for the index build at all.
    root, cfg, runtime = env
    _seed_search_index(root, [])

    def _boom(_cfg: ServerConfig) -> tools_read_mod._EntityIndex:
        raise AssertionError("build_entity_index must not run when there are no hits")

    monkeypatch.setattr(tools_read_mod, "build_entity_index", _boom)

    out = tool_search(cfg, runtime, query="nothing-matches-anything", mode="lexical", top_k=10)
    assert out.hits == []


def test_build_entity_index_slug_exact_match_beats_probable(env: _Env) -> None:
    # F3 + F4: a hit whose title slugifies to exactly the same slug as an
    # entity's own title (spaces vs hyphens) is "exists", never "probable".
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/dates/dates.md", title="2026-07-21")

    idx = tools_read_mod.build_entity_index(cfg)
    hit = idx.hint("archive/processed/x.md", "2026 07 21")
    assert hit.status == "exists"
    assert hit.node_id == "projects/dates/dates"


def test_build_entity_index_fragmentation_rule_ignores_short_and_dated_slugs(
    env: _Env,
) -> None:
    # F3: the minimum slug length (8) keeps short slugs from fuzzy-matching
    # each other (a real vault false positive: "Regex"/"Rox" against "Rex"),
    # and dated project log entries are excluded from the index entirely
    # (F2), so a bare date title never resolves against a project.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/rex/rex.md", title="Rex")
    _write_entity_note(root, "knowledge/projects/claude/claude.md", title="Claude")
    _write_entity_note(
        root, "knowledge/projects/claude/log/2026-07-20.md", title="2026-07-20",
    )

    idx = tools_read_mod.build_entity_index(cfg)
    assert idx.hint("archive/processed/x.md", "Regex").status == "unknown"
    assert idx.hint("archive/processed/x.md", "Rox").status == "unknown"
    assert idx.hint("archive/processed/x.md", "2026 07 21").status == "unknown"


def test_build_entity_index_fragmentation_rule_still_catches_real_near_miss(
    env: _Env,
) -> None:
    # F3: the tightened rule (distance <= 1, min length 8) still catches
    # the genuine near-miss it was ported from sweep to catch.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/olympus/olympus.md", title="Project Olympus")

    idx = tools_read_mod.build_entity_index(cfg)
    hit = idx.hint("archive/processed/x.md", "projectolympus")
    assert hit.status == "probable"
    assert hit.node_id == "projects/olympus/olympus"


def test_build_entity_index_probable_ties_broken_by_node_id(env: _Env) -> None:
    # F4: scores every candidate and returns the nearest; among equal-
    # distance candidates, the lower node id wins, not sort/insertion order.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/bbb/bbb.md", title="Abcdefgi")
    _write_entity_note(root, "knowledge/projects/aaa/aaa.md", title="Abcdefgh")

    idx = tools_read_mod.build_entity_index(cfg)
    hit = idx.hint("archive/processed/x.md", "Abcdefgx")
    assert hit.status == "probable"
    assert hit.node_id == "projects/aaa/aaa"


def test_build_entity_index_keeps_a_projects_curated_notes_as_their_own_nodes(
    env: _Env,
) -> None:
    # N1: AGENTS.md — "curated notes beside it keep their own path" — and 36
    # of the live vault's stored relation targets ARE such notes. Folding
    # them into their project made the hint answer with the parent project,
    # so an agent following it linked the wrong node; a title lookup for one
    # answered "unknown", so an agent minted a duplicate instead.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/fps/fps.md", title="FPS")
    _write_entity_note(
        root, "knowledge/projects/fps/stack-and-architecture.md",
        title="Stack and architecture",
    )
    _write_entity_note(
        root, "knowledge/projects/fps/log/2026-07-20.md", title="2026-07-20",
    )

    idx = tools_read_mod.build_entity_index(cfg)

    curated = idx.hint("knowledge/projects/fps/stack-and-architecture.md", "whatever")
    assert curated.status == "exists"
    assert curated.node_id == "projects/fps/stack-and-architecture"

    by_title = idx.hint("archive/processed/x.md", "Stack and architecture")
    assert by_title.status == "exists"
    assert by_title.node_id == "projects/fps/stack-and-architecture"

    # The dated log entry is still not a node of its own.
    log_hit = idx.hint("knowledge/projects/fps/log/2026-07-20.md", "2026-07-20")
    assert log_hit.status == "exists"
    assert log_hit.node_id == "projects/fps/fps"


def test_build_entity_index_names_a_flat_project_note(env: _Env) -> None:
    # N2: a flat knowledge/projects/<slug>.md is a shape AGENTS.md
    # contemplates and one is live; answering "unknown" for it is how an
    # agent is told to mint a duplicate of a project that already exists.
    root, cfg, runtime = env
    _write_entity_note(
        root, "knowledge/projects/zymbly-role.md", title="Zymbly role",
    )
    idx = tools_read_mod.build_entity_index(cfg)

    hit = idx.hint("knowledge/projects/zymbly-role.md", "Zymbly role")
    assert hit.status == "exists"
    assert hit.node_id == "projects/zymbly-role"


def test_build_entity_index_ignores_entities_differing_only_in_numbers(
    env: _Env,
) -> None:
    # N3(a): "retro-20260101" vs "retro-20260102" is a dated edition pair,
    # never one entity spelled twice — the rule ported from sweep. Without
    # it the near-miss scan calls every dated title a probable match for
    # the previous one.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/projects/retro/retro.md", title="Retro 20260101")

    idx = tools_read_mod.build_entity_index(cfg)
    assert idx.hint("archive/processed/x.md", "Retro 20260102").status == "unknown"
    # Control: the same slug shape with the digits changed to letters is a
    # genuine near-miss, so the scan itself is live.
    _write_entity_note(root, "knowledge/projects/letters/letters.md", title="Retro abcdefgh")
    idx = tools_read_mod.build_entity_index(cfg)
    assert idx.hint("archive/processed/x.md", "Retro abcdefgx").status == "probable"


def test_build_entity_index_probable_picks_the_nearest_not_the_first(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # N3(b): with the shipped budget every admitted candidate is exactly one
    # edit away, so nothing distinguishes "nearest wins" from "whichever
    # sorts first" — inverting the scorer left the suite green. Widening the
    # budget for this test alone is what puts two DIFFERENT distances in
    # front of the scorer; the node ids are chosen so alphabetical order and
    # farthest-first both pick the wrong one.
    root, cfg, runtime = env
    monkeypatch.setattr(tools_read_mod, "_ENTITY_NEAR_MAX_DISTANCE", 2)
    _write_entity_note(root, "knowledge/projects/zzz/zzz.md", title="abcdefgx")
    _write_entity_note(root, "knowledge/projects/aaa/aaa.md", title="abcdefxy")

    idx = tools_read_mod.build_entity_index(cfg)
    hit = idx.hint("archive/processed/x.md", "abcdefgh")
    assert hit.status == "probable"
    assert hit.node_id == "projects/zzz/zzz"


def test_tool_search_builds_the_entity_index_once_for_all_hits(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # N3(c): the index docstring promises "built once per search call (never
    # once per hit)" — the whole reason the per-hit lookups are cheap.
    # Rebuilding it per hit passed every existing test.
    root, cfg, runtime = env
    _write_entity_note(root, "knowledge/people/anna-kowalska.md", title="Anna Kowalska")
    for name in ("note-a", "note-b", "note-c"):
        _write_note(root, f"knowledge/notes/{name}.md")
    _seed_search_index(root, [
        {
            "source_relative_path": f"knowledge/notes/{name}.md",
            "source_hash": f"h{i}", "title": "Anna Kowalska", "chunk_idx": 0,
            "text": "congestion body", "origin": "knowledge-note",
        }
        for i, name in enumerate(("note-a", "note-b", "note-c"))
    ])

    real_build = tools_read_mod.build_entity_index
    builds = 0

    def _counting(cfg_arg: ServerConfig) -> tools_read_mod._EntityIndex:
        nonlocal builds
        builds += 1
        return real_build(cfg_arg)

    monkeypatch.setattr(tools_read_mod, "build_entity_index", _counting)

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)
    assert len(out.hits) == 3
    assert {h.evidence.node_id for h in out.hits} == {"people/anna-kowalska"}
    assert builds == 1


# ----------------------------------------------------------- tool_related

def _rec(path: str, topics: list[str]) -> IndexRecord:
    return IndexRecord(
        relative_path=path, source_hash="h-" + path, size_bytes=1, extension=".md",
        extractor="text", status="processed", raw_path="archive/raw/" + path,
        processed_path="archive/processed/" + path, index_note_path=None, topics=topics,
    )


def test_tool_related_returns_ranked_neighbours(env: _Env) -> None:
    root, cfg, runtime = env
    for rec in (
        _rec("a.md", ["Alpha", "Beta"]),
        _rec("b.md", ["Beta", "Gamma"]),
        _rec("c.md", ["Alpha", "Beta", "Gamma"]),
    ):
        append_record(root / "metadata" / "index.jsonl", rec)
    rebuild_connections(paths_for_root(root), logger=_LOG)

    out = tool_related(cfg, runtime, concept="Beta", limit=8)

    assert out.concept == "beta"
    assert {r.slug for r in out.related} == {"alpha", "gamma"}

    access_path = root / "logs" / "mcp-access.jsonl"
    rows = [json.loads(ln) for ln in access_path.read_text(encoding="utf-8").splitlines()]
    assert any(r["tool"] == "vault_related" and r["query"] == "Beta" for r in rows)


def test_tool_related_unknown_concept_refuses(env: _Env) -> None:
    root, cfg, runtime = env
    append_record(root / "metadata" / "index.jsonl", _rec("a.md", ["Alpha"]))
    from ingest_lib.config import paths_for_root
    rebuild_connections(paths_for_root(root), logger=_LOG)
    with pytest.raises(ToolError, match="unknown concept"):
        tool_related(cfg, runtime, concept="not-a-topic")


# -------------------------------------------------------------- tool_list

def test_tool_list_hides_children_the_read_policy_denies(env: _Env) -> None:
    root, cfg, runtime = env
    (root / "knowledge").mkdir()
    (root / "secrets.txt").write_text("shh", encoding="utf-8")

    out = tool_list(cfg, runtime, path="")

    names = {e.name: e for e in out.entries}
    assert "knowledge" in names and names["knowledge"].is_dir
    assert "secrets.txt" not in names


def test_tool_list_lists_files_with_sizes_under_a_subdir(env: _Env) -> None:
    root, cfg, runtime = env
    notes = root / "knowledge" / "notes"
    notes.mkdir(parents=True)
    (notes / "a.md").write_text("hello", encoding="utf-8")

    out = tool_list(cfg, runtime, path="knowledge/notes")

    assert out.path == "knowledge/notes"
    assert len(out.entries) == 1
    assert out.entries[0].name == "a.md"
    assert out.entries[0].size_bytes == 5


# ---------------------------------------------------- tool_metadata_query

def test_tool_metadata_query_filters_by_status_extension_and_prefix(env: _Env) -> None:
    root, cfg, runtime = env
    jsonl = root / "metadata" / "index.jsonl"
    append_record(jsonl, IndexRecord(
        relative_path="uni/lecture1.pdf", source_hash="h1", size_bytes=10,
        extension=".pdf", extractor="pdf-mineru", status="processed",
        raw_path="archive/raw/uni/lecture1.pdf",
        processed_path="archive/processed/uni/lecture1.md", index_note_path=None,
    ))
    append_record(jsonl, IndexRecord(
        relative_path="uni/scan.pdf", source_hash="h2", size_bytes=20,
        extension=".pdf", extractor="pdf-pypdf", status="partial",
        raw_path="archive/raw/uni/scan.pdf",
        processed_path="archive/processed/uni/scan.md", index_note_path=None,
    ))
    append_record(jsonl, IndexRecord(
        relative_path="notes/x.txt", source_hash="h3", size_bytes=5,
        extension=".txt", extractor="text", status="processed",
        raw_path="archive/raw/notes/x.txt",
        processed_path="archive/processed/notes/x.md", index_note_path=None,
    ))

    by_status = tool_metadata_query(cfg, runtime, by="status", value="partial")
    assert [r.relative_path for r in by_status.records] == ["uni/scan.pdf"]

    by_ext = tool_metadata_query(cfg, runtime, by="extension", value=".txt")
    assert [r.relative_path for r in by_ext.records] == ["notes/x.txt"]

    by_prefix = tool_metadata_query(cfg, runtime, by="path_prefix", value="uni/")
    assert {r.relative_path for r in by_prefix.records} == {"uni/lecture1.pdf", "uni/scan.pdf"}

    everything = tool_metadata_query(cfg, runtime, by="all")
    assert len(everything.records) == 3


def test_tool_metadata_query_rejects_bad_by_and_missing_value(env: _Env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="by must be one of"):
        tool_metadata_query(cfg, runtime, by="bogus")
    with pytest.raises(ToolError, match="value is required"):
        tool_metadata_query(cfg, runtime, by="status", value=None)


# ------------------------------------------------------- tool_drop_inbox_file

def test_tool_drop_inbox_file_writes_bytes_and_refuses_overwrite(env: _Env) -> None:
    root, cfg, runtime = env
    payload = b"%PDF-1.4 fake pdf bytes for a drop test"
    b64 = base64.b64encode(payload).decode()

    res = tool_drop_inbox_file(cfg, runtime, path="paper.pdf", content_base64=b64)

    assert res.committed and res.commit_sha
    dropped = root / "inbox" / "paper.pdf"
    assert dropped.read_bytes() == payload

    log_line = _git(root, "log", "-1", "--format=%s")
    assert log_line == "mcp(agent-a): drop inbox file paper.pdf"

    with pytest.raises(ToolError, match="already exists"):
        tool_drop_inbox_file(cfg, runtime, path="paper.pdf", content_base64=b64)
    # The first drop's bytes must survive the refused second attempt.
    assert dropped.read_bytes() == payload


def test_tool_search_drops_hits_for_notes_deleted_off_disk(env: _Env) -> None:
    # AUD-050: there is no MCP delete verb, so a note deleted in Obsidian
    # keeps its index rows until the next rebuild. The read gate now checks
    # the note still exists rather than serving the ghost.
    root, cfg, runtime = env
    _write_note(root, "knowledge/notes/live.md")
    _seed_search_index(root, [
        {
            "source_relative_path": "knowledge/notes/live.md", "source_hash": "h1",
            "title": "live", "chunk_idx": 0, "text": "congestion window",
            "origin": "knowledge-note",
        },
        {
            "source_relative_path": "knowledge/notes/ghost.md", "source_hash": "h2",
            "title": "ghost", "chunk_idx": 0, "text": "congestion window",
            "origin": "knowledge-note",
        },
    ])

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)

    assert [h.source_relative_path for h in out.hits] == ["knowledge/notes/live.md"]


def test_tool_search_keeps_archive_hits_whose_derived_twin_is_absent(env: _Env) -> None:
    # The archive gate path is DERIVED, not read from the record: vaults
    # built before the extension-keeping convention hold the note at the old
    # <source>.md path. Existence there is a naming signal, not a deletion
    # signal, so those hits must survive the AUD-050 check.
    root, cfg, runtime = env
    _seed_search_index(root, [
        {
            "source_relative_path": "uni/lecture.pdf", "source_hash": "h1",
            "title": "lecture", "chunk_idx": 0, "text": "congestion window",
            "origin": "pdf-mineru",
        },
    ])
    assert not (root / "archive" / "processed" / "uni" / "lecture.pdf.md").exists()

    out = tool_search(cfg, runtime, query="congestion", mode="lexical", top_k=10)

    assert [h.source_relative_path for h in out.hits] == ["uni/lecture.pdf"]


def test_tool_metadata_query_pages_with_offset(env: _Env) -> None:
    # AUD-049: without an offset an agent could never enumerate past `limit`.
    root, cfg, runtime = env
    jsonl = root / "metadata" / "index.jsonl"
    for i in range(5):
        append_record(jsonl, IndexRecord(
            relative_path=f"uni/l{i}.pdf", source_hash=f"h{i}", size_bytes=1,
            extension=".pdf", extractor="pdf-mineru", status="processed",
            raw_path=f"archive/raw/uni/l{i}.pdf",
            processed_path=None, index_note_path=None,
        ))

    first = tool_metadata_query(cfg, runtime, by="all", limit=2)
    second = tool_metadata_query(cfg, runtime, by="all", limit=2, offset=2)
    tail = tool_metadata_query(cfg, runtime, by="all", limit=2, offset=4)

    assert [r.relative_path for r in first.records] == ["uni/l0.pdf", "uni/l1.pdf"]
    assert [r.relative_path for r in second.records] == ["uni/l2.pdf", "uni/l3.pdf"]
    assert [r.relative_path for r in tail.records] == ["uni/l4.pdf"]
    with pytest.raises(ToolError, match="offset"):
        tool_metadata_query(cfg, runtime, by="all", offset=-1)


def test_tool_metadata_query_includes_knowledge_notes(env: _Env) -> None:
    # AUD-049: vault_search labels hand-edited notes `knowledge-note`, but
    # they live nowhere in index.jsonl, so the metadata view used to answer
    # "nothing" for the extractor its sibling reports.
    root, cfg, runtime = env
    _write_note(root, "knowledge/notes/tcp.md")
    append_record(root / "metadata" / "index.jsonl", IndexRecord(
        relative_path="uni/lecture.pdf", source_hash="h1", size_bytes=1,
        extension=".pdf", extractor="pdf-mineru", status="processed",
        raw_path="archive/raw/uni/lecture.pdf",
        processed_path=None, index_note_path=None,
    ))

    notes = tool_metadata_query(cfg, runtime, by="extractor", value="knowledge-note")
    assert [r.relative_path for r in notes.records] == ["knowledge/notes/tcp.md"]

    everything = tool_metadata_query(cfg, runtime, by="all")
    assert [r.relative_path for r in everything.records] == [
        "knowledge/notes/tcp.md", "uni/lecture.pdf",
    ]
