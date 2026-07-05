"""Tests for the structured entity write tools (B1b).

Same harness as test_mcp_provenance: a throwaway git vault, a real
Runtime with both background workers disabled, AGENT_VAR pinned to a
test agent. Fully offline — no model loads, no network, no pushes.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, UTC
from pathlib import Path

import pytest

from mcp_server import tools as tools_mod
from mcp_server.audit import AuditLog
from mcp_server.config import ServerConfig
from mcp_server.entity_tools import (
    tool_entity_append_fact,
    tool_entity_upsert_relation,
    tool_meeting_create,
    tool_relations_query,
)
from mcp_server.identity import AGENT_VAR
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import ToolError


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


def _entity(title: str, type_: str) -> str:
    return (
        "---\n"
        f"title: '{title}'\n"
        f"type: {type_}\n"
        "topics: []\n"
        "relations: []\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        "## Notes\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Log\n"
        "\n"
        "_(empty)_\n"
    )


def _seed(root: Path) -> None:
    notes = {
        "knowledge/people/anna.md": _entity("Anna Kowalska", "person"),
        "knowledge/people/bob.md": _entity("Bob Smith", "person"),
        "knowledge/organisations/acme.md": _entity("Acme", "organisation"),
        "knowledge/projects/fyp.md": _entity("FYP", "project"),
        "knowledge/meetings/2026/2026-06-01-kickoff.md": _entity("Kickoff", "meeting"),
    }
    for rel, text in notes.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _roomy_rate_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    # The write bucket (30/min) is module-global and shared by the whole
    # pytest process; a fast full-suite run would otherwise eat the budget
    # across files. Fresh roomy buckets keep tests order-independent.
    monkeypatch.setattr(tools_mod, "_write_bucket", tools_mod._RateBucket(10_000))


@pytest.fixture()
def env(tmp_path: Path):
    root = _make_vault(tmp_path)
    _seed(root)
    token = AGENT_VAR.set("agent-a")
    try:
        yield root, _cfg(root), _runtime(root)
    finally:
        AGENT_VAR.reset(token)


SOURCE = "knowledge/meetings/2026/2026-06-01-kickoff"


# ------------------------------------------------- entity_upsert_relation

def test_upsert_adds_relation(env) -> None:
    root, cfg, runtime = env
    res = tool_entity_upsert_relation(
        cfg, runtime,
        entity_path="knowledge/people/anna.md",
        rel="works_at", target="organisations/acme",
        valid_from="2026-01-01", source=SOURCE,
    )
    assert res.action == "added"
    assert res.committed and res.commit_sha
    assert res.path == "knowledge/people/anna.md"
    on_disk = (root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    assert "rel: works_at" in on_disk
    assert "target: organisations/acme" in on_disk
    assert "valid_from: '2026-01-01'" in on_disk
    assert f"source: {SOURCE}" in on_disk
    assert "last_written_by: 'agent:agent-a'" in on_disk
    # Body survives the frontmatter rewrite byte-for-byte.
    assert on_disk.endswith("## Log\n\n_(empty)_\n")
    assert _git(root, "log", "-1", "--format=%s") == \
        "mcp(agent-a): relation works_at people/anna -> organisations/acme"


def test_upsert_closes_open_relation(env) -> None:
    root, cfg, runtime = env
    tool_entity_upsert_relation(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        rel="works_at", target="organisations/acme", valid_from="2026-01-01",
    )
    res = tool_entity_upsert_relation(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        rel="works_at", target="organisations/acme", valid_until="2026-05-01",
    )
    assert res.action == "closed"
    on_disk = (root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    assert "valid_until: '2026-05-01'" in on_disk
    # Closed in place: still exactly one works_at entry.
    assert on_disk.count("rel: works_at") == 1


def test_upsert_noop_skips_write_and_commit(env) -> None:
    root, cfg, runtime = env
    tool_entity_upsert_relation(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        rel="works_at", target="organisations/acme",
    )
    sha = _git(root, "rev-parse", "HEAD")
    before = (root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    # Same relation again, in wikilink form — normalisation makes it match.
    res = tool_entity_upsert_relation(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        rel="works_at", target="[[knowledge/organisations/acme]]",
    )
    assert res.action == "noop"
    assert res.warning == "relation already present"
    assert not res.committed and res.commit_sha is None
    assert res.push_state == "skipped" and res.index_refresh == "skipped"
    assert (root / "knowledge/people/anna.md").read_text(encoding="utf-8") == before
    assert _git(root, "rev-parse", "HEAD") == sha  # no new commit


def test_upsert_rejects_unknown_rel(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="works_at"):  # vocabulary is listed
        tool_entity_upsert_relation(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            rel="employed_by", target="organisations/acme",
        )


def test_upsert_rejects_missing_target(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match=r"knowledge/organisations/ghost\.md"):
        tool_entity_upsert_relation(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            rel="works_at", target="organisations/ghost",
        )


def test_upsert_rejects_nonexistent_entity(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="vault_create_note"):
        tool_entity_upsert_relation(
            cfg, runtime, entity_path="knowledge/people/ghost.md",
            rel="works_at", target="organisations/acme",
        )


@pytest.mark.parametrize("field,kwargs", [
    ("valid_from", {"valid_from": "01-01-2026"}),
    ("valid_from", {"valid_from": "2026-1-1"}),     # non-canonical
    ("valid_until", {"valid_until": "2026-13-40"}),  # impossible
])
def test_upsert_rejects_bad_dates(env, field: str, kwargs: dict) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        tool_entity_upsert_relation(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            rel="works_at", target="organisations/acme", **kwargs,
        )


# ---------------------------------------------------- entity_append_fact

def test_append_fact_happy_path(env) -> None:
    root, cfg, runtime = env
    res = tool_entity_append_fact(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        text="Prefers async standups", source=SOURCE, date="2026-06-01",
    )
    assert res.committed and res.commit_sha
    on_disk = (root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    bullet = f"- 2026-06-01 — Prefers async standups ([[{SOURCE}]])"
    assert bullet in on_disk
    # The bullet landed inside the Log section, not appended after EOF junk.
    assert on_disk.index("## Log") < on_disk.index(bullet)
    assert "last_written_by: 'agent:agent-a'" in on_disk
    assert _git(root, "log", "-1", "--format=%s") == "mcp(agent-a): fact -> people/anna"


def test_append_fact_defaults_date_to_today_utc(env) -> None:
    root, cfg, runtime = env
    # Capture the date window AROUND the call so a run that crosses UTC
    # midnight between the tool stamp and this assertion doesn't flake.
    before = datetime.now(UTC).date().isoformat()
    tool_entity_append_fact(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        text="Joined the platform guild", source=SOURCE,
    )
    after = datetime.now(UTC).date().isoformat()
    on_disk = (root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    assert any(
        f"- {d} — Joined the platform guild ([[{SOURCE}]])" in on_disk
        for d in {before, after}
    )


@pytest.mark.parametrize("bad_text,expected", [
    ("line one\nline two", "single line"),
    ("", "non-empty"),
    ("   ", "non-empty"),
    ("x" * 501, "500"),
])
def test_append_fact_rejects_bad_text(env, bad_text: str, expected: str) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match=expected):
        tool_entity_append_fact(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            text=bad_text, source=SOURCE,
        )


def test_append_fact_requires_source(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="source is required"):
        tool_entity_append_fact(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            text="A fact", source="",
        )


def test_append_fact_rejects_missing_source_note(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="does not exist"):
        tool_entity_append_fact(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            text="A fact", source="knowledge/meetings/2026/2026-06-02-ghost",
        )


def test_append_fact_rejects_bad_date(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        tool_entity_append_fact(
            cfg, runtime, entity_path="knowledge/people/anna.md",
            text="A fact", source=SOURCE, date="June 1st",
        )


# --------------------------------------------------------- meeting_create

def test_meeting_create_happy_path(env) -> None:
    root, cfg, runtime = env
    res = tool_meeting_create(
        cfg, runtime,
        date="2026-06-12", title="Kern Call",
        # Mixed id forms: normalisation must accept the wikilink spelling.
        attendees=["people/anna", "[[knowledge/people/bob]]"],
        project="projects/fyp",
        body="We discussed scope.",
    )
    rel = "knowledge/meetings/2026/2026-06-12-kern-call.md"
    assert res.path == rel
    assert res.committed and res.commit_sha

    meeting = (root / rel).read_text(encoding="utf-8")
    assert "type: meeting" in meeting
    assert "'2026-06-12'" in meeting          # date, single-quoted
    assert "people/anna" in meeting and "people/bob" in meeting
    assert "topics: []" in meeting
    assert "created:" in meeting and "updated:" in meeting
    # The project edge lives in the meeting's OWN frontmatter.
    assert "rel: related_to" in meeting
    assert "target: projects/fyp" in meeting
    # Body skeleton + wikilinks (full vault-relative, no extension).
    assert "# Kern Call" in meeting
    for heading in ("## Agenda", "## Notes", "## Decisions", "## Actions", "## Links"):
        assert heading in meeting
    assert "We discussed scope." in meeting
    assert "[[knowledge/people/anna]]" in meeting
    assert "[[knowledge/people/bob]]" in meeting
    assert "[[knowledge/projects/fyp]]" in meeting
    assert "author: 'agent:agent-a'" in meeting

    # Every attendee gained the attended relation, with provenance.
    for person in ("anna", "bob"):
        note = (root / f"knowledge/people/{person}.md").read_text(encoding="utf-8")
        assert "rel: attended" in note
        assert "target: meetings/2026/2026-06-12-kern-call" in note
        assert "valid_from: '2026-06-12'" in note
        assert "source: knowledge/meetings/2026/2026-06-12-kern-call" in note
        assert "last_written_by: 'agent:agent-a'" in note

    # ONE commit covering the meeting note and both attendee notes.
    assert _git(root, "log", "-1", "--format=%s") == \
        "mcp(agent-a): meeting 2026-06-12-kern-call"
    changed = set(_git(root, "show", "--name-only", "--format=", "HEAD").splitlines())
    assert changed == {rel, "knowledge/people/anna.md", "knowledge/people/bob.md"}


def test_meeting_create_lists_all_missing_attendees(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError) as exc:
        tool_meeting_create(
            cfg, runtime, date="2026-06-12", title="Ghost Sync",
            attendees=["people/ghost-one", "people/anna", "people/ghost-two"],
        )
    msg = str(exc.value)
    assert "knowledge/people/ghost-one.md" in msg
    assert "knowledge/people/ghost-two.md" in msg
    assert "knowledge/people/anna.md" not in msg  # exists; not reported


def test_meeting_create_refuses_duplicate(env) -> None:
    _root, cfg, runtime = env
    tool_meeting_create(
        cfg, runtime, date="2026-06-12", title="Kern Call",
        attendees=["people/anna"],
    )
    with pytest.raises(ToolError, match="already exists"):
        tool_meeting_create(
            cfg, runtime, date="2026-06-12", title="Kern Call",
            attendees=["people/anna"],
        )


def test_meeting_create_without_project_has_no_relation(env) -> None:
    root, cfg, runtime = env
    tool_meeting_create(
        cfg, runtime, date="2026-06-12", title="No Project Standup",
        attendees=["people/anna"],
    )
    meeting = (root / "knowledge/meetings/2026/2026-06-12-no-project-standup.md") \
        .read_text(encoding="utf-8")
    assert "related_to" not in meeting
    assert "project: ''" in meeting
    assert "_(empty)_" in meeting  # no body -> placeholder under Notes


def test_meeting_create_rejects_missing_project(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match=r"knowledge/projects/ghost\.md"):
        tool_meeting_create(
            cfg, runtime, date="2026-06-12", title="Kern Call",
            attendees=["people/anna"], project="projects/ghost",
        )


def test_meeting_create_is_all_or_nothing(env) -> None:
    root, cfg, runtime = env
    anna_before = (root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    with pytest.raises(ToolError, match="missing attendee"):
        tool_meeting_create(
            cfg, runtime, date="2026-06-12", title="Half Done",
            attendees=["people/anna", "people/ghost"],
        )
    # Failed validation left ZERO writes behind.
    assert not (root / "knowledge/meetings/2026/2026-06-12-half-done.md").exists()
    assert (root / "knowledge/people/anna.md").read_text(encoding="utf-8") == anna_before
    # And no commit was made at all (repo still has no HEAD).
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


@pytest.mark.parametrize("kwargs,expected", [
    ({"date": "12/06/2026", "title": "X", "attendees": ["people/anna"]}, "YYYY-MM-DD"),
    ({"date": "2026-06-12", "title": "   ", "attendees": ["people/anna"]}, "non-empty"),
    ({"date": "2026-06-12", "title": "!!!", "attendees": ["people/anna"]}, "slugifies"),
    ({"date": "2026-06-12", "title": "X", "attendees": []}, "at least one"),
    ({"date": "2026-06-12", "title": "X", "attendees": ["organisations/acme"]},
     "people/ node id"),
])
def test_meeting_create_validates_inputs(env, kwargs: dict, expected: str) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match=expected):
        tool_meeting_create(cfg, runtime, **kwargs)


# ------------------------------------------------- relations_query (P3)

def test_relations_query_reverse_and_as_of(env) -> None:
    root, cfg, runtime = env
    # Anna: at Acme 2025, moved to Initech 2026 (seed has anna + acme; add initech).
    (root / "knowledge/organisations/initech.md").write_text(
        "---\ntitle: Initech\n---\n", encoding="utf-8")
    tool_entity_upsert_relation(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        rel="works_at", target="organisations/acme",
        valid_from="2025-01-01", valid_until="2026-01-01", source=SOURCE)
    tool_entity_upsert_relation(
        cfg, runtime, entity_path="knowledge/people/anna.md",
        rel="works_at", target="organisations/initech",
        valid_from="2026-01-01", source=SOURCE)

    # Default: only the open relation.
    out = tool_relations_query(cfg, runtime, entity="people/anna")
    assert [(h.rel, h.target) for h in out.relations] == [("works_at", "organisations/initech")]

    # as_of mid-2025 -> the historical Acme interval.
    hist = tool_relations_query(cfg, runtime, entity="people/anna", as_of="2025-06-01")
    assert [h.target for h in hist.relations] == ["organisations/acme"]

    # Reverse lookup by target.
    rev = tool_relations_query(cfg, runtime, target="organisations/initech")
    assert [h.entity for h in rev.relations] == ["people/anna"]


def test_relations_query_rejects_bad_rel_and_date(env) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="unknown rel"):
        tool_relations_query(cfg, runtime, rel="employed_by")
    with pytest.raises(ToolError, match="as_of"):
        tool_relations_query(cfg, runtime, as_of="2026/01/01")
