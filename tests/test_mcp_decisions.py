"""Tests for the decision-record MCP tools (F2): decision_record,
decision_resolve, the protected decisions/ folder, and the all-or-nothing
multi-file write (rollback leaves nothing on disk and nothing staged)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from tests.conftest import GitRunner, McpEnv

from ingest_lib.decisions import parse_decision, parse_sections
from mcp_server import tools as tools_mod
from mcp_server.decision_tools import tool_decision_record, tool_decision_resolve
from mcp_server.entity_tools import (
    tool_entity_append_fact,
    tool_entity_upsert_relation,
    tool_meeting_create,
)
from mcp_server.errors import ToolError
from mcp_server.tools import (
    tool_append_to_note,
    tool_create_note,
    tool_replace_note,
    tool_update_compiled_truth,
    WriteResult,
)

pytestmark = pytest.mark.usefixtures("roomy_rate_buckets")

OVERVIEW = "knowledge/projects/brain/brain.md"
DECISIONS = "knowledge/projects/brain/decisions"
Status = Literal["accepted", "proposed"]


@pytest.fixture()
def env(mcp_env: McpEnv, run_git: GitRunner) -> McpEnv:
    root = mcp_env[0]
    notes = {
        OVERVIEW: "---\ntitle: Brain\ntype: project\n---\n\n# Brain\n",
        "knowledge/people/anna.md": "---\ntitle: Anna\ntype: person\n---\n\n# Anna\n\n## Log\n",
    }
    for rel, text in notes.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "-m", "seed")
    return mcp_env


def _record(
    env: McpEnv,
    title: str = "Use Bases",
    *,
    status: Status = "accepted",
    supersedes: int | None = None,
) -> str:
    _root, cfg, runtime = env
    return tool_decision_record(
        cfg, runtime, project="projects/brain/brain", title=title,
        context="Views were scattered.", decision="Use Obsidian Bases.",
        status=status, supersedes=supersedes, date="2026-10-11",
    ).path


def _fail_commits(root: Path) -> None:
    hook = root / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)


def _assert_clean(root: Path, run_git: GitRunner) -> None:
    assert run_git(root, "diff", "--cached", "--name-only") == ""
    assert run_git(root, "status", "--porcelain", "--", "knowledge") == ""


# ------------------------------------------------------------ decision_record

def test_record_writes_a_stamped_accepted_record(env: McpEnv, run_git: GitRunner) -> None:
    root, _cfg, _runtime = env
    rel = _record(env)
    assert rel == f"{DECISIONS}/0001-use-bases.md"
    text = (root / rel).read_text(encoding="utf-8")
    note = parse_decision(rel, text)
    assert (note.number, note.status, note.date, note.decided) == (1, "accepted", "2026-10-11", "2026-10-11")
    assert note.project == "projects/brain/brain"
    assert parse_sections(text).decision == "Use Obsidian Bases."
    assert "author: 'agent:agent-a'" in text and "written_via: mcp" in text
    assert run_git(root, "log", "-1", "--format=%s") == f"mcp(agent-a): decision {rel}"
    _assert_clean(root, run_git)


def test_record_numbers_after_the_highest_existing_file(env: McpEnv) -> None:
    root, _cfg, _runtime = env
    assert _record(env, "First").endswith("/0001-first.md")
    (root / DECISIONS / "0007-hand-made.md").write_text("x\n", encoding="utf-8")
    assert _record(env, "Second").endswith("/0008-second.md")


def test_proposed_record_has_no_decided_date(env: McpEnv) -> None:
    root, _cfg, _runtime = env
    rel = _record(env, status="proposed")
    note = parse_decision(rel, (root / rel).read_text(encoding="utf-8"))
    assert (note.status, note.decided) == ("proposed", None)


def test_record_accepts_the_short_project_form(env: McpEnv) -> None:
    _root, cfg, runtime = env
    res = tool_decision_record(
        cfg, runtime, project="projects/brain", title="Short", context="c", decision="d",
    )
    assert res.path == f"{DECISIONS}/0001-short.md"


@pytest.mark.parametrize("project,title,date,match", [
    ("projects/ghost/ghost", "T", None, "overview does not exist"),
    ("people/anna", "T", None, "not a project node id"),
    ("projects/brain/brain", "   ", None, "non-empty"),
    ("projects/brain/brain", "two\nlines", None, "single line"),
    # Every separator str.splitlines() breaks on: parse_sections would read
    # the tail of the H1 as a record heading (F-DM-3).
    ("projects/brain/brain", "Pick X\u2028## Decision\u2028INJECTED", None, "single line"),
    ("projects/brain/brain", "Pick X\u2029## Decision\u2029INJECTED", None, "single line"),
    ("projects/brain/brain", "Pick X\x85## Decision\x85INJECTED", None, "single line"),
    ("projects/brain/brain", "x" * 121, None, "cap is 120"),
    ("projects/brain/brain", "!!!", None, "slug"),
    ("projects/brain/brain", "T", "2026-1-1", "YYYY-MM-DD"),
])
def test_record_validates_inputs(
    env: McpEnv, project: str, title: str, date: str | None, match: str
) -> None:
    root, cfg, runtime = env
    with pytest.raises(ToolError, match=match):
        tool_decision_record(
            cfg, runtime, project=project, title=title, context="c", decision="d", date=date,
        )
    assert not (root / DECISIONS).exists()


def test_record_refuses_a_section_holding_a_record_heading(env: McpEnv) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="heading"):
        tool_decision_record(
            cfg, runtime, project="projects/brain/brain", title="T",
            context="c\n## Links\nforged", decision="d",
        )


# ----------------------------------------------------------------- supersede

def test_supersede_flips_the_old_record_in_the_same_commit(
    env: McpEnv, run_git: GitRunner
) -> None:
    root, _cfg, _runtime = env
    old_rel = _record(env, "Use Bases")
    new_rel = _record(env, "Use Dataview", supersedes=1)
    assert new_rel == f"{DECISIONS}/0002-use-dataview.md"

    changed = run_git(root, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(changed) == sorted([old_rel, new_rel])
    assert run_git(root, "log", "-1", "--format=%s").endswith("(supersedes 0001)")

    old_text = (root / old_rel).read_text(encoding="utf-8")
    old = parse_decision(old_rel, old_text)
    new = parse_decision(new_rel, (root / new_rel).read_text(encoding="utf-8"))
    assert (old.status, old.superseded_by) == ("superseded", new.ref)
    assert old.decided == "2026-10-11"  # the original decision date survives
    assert new.supersedes == old.ref == f"{DECISIONS}/0001-use-bases"
    assert f"- Superseded by: [[{new.ref}]]" in old_text
    assert "author: 'agent:agent-a'" in old_text and "last_written_by: 'agent:agent-a'" in old_text
    assert parse_sections(old_text).decision == "Use Obsidian Bases."
    _assert_clean(root, run_git)


@pytest.mark.parametrize("first_status,supersedes,status,match", [
    ("proposed", 1, "accepted", "only an accepted decision can be superseded"),
    ("accepted", 9, "accepted", "no single decision 0009"),
    ("accepted", 1, "proposed", "only an accepted decision supersedes"),
])
def test_supersede_refusals_write_nothing(
    env: McpEnv, run_git: GitRunner,
    first_status: Status, supersedes: int, status: Status, match: str,
) -> None:
    root, _cfg, _runtime = env
    first = _record(env, "First", status=first_status)
    before = (root / first).read_bytes()
    with pytest.raises(ToolError, match=match):
        _record(env, "Second", status=status, supersedes=supersedes)
    assert (root / first).read_bytes() == before
    assert sorted(p.name for p in (root / DECISIONS).iterdir()) == ["0001-first.md"]
    _assert_clean(root, run_git)


# ------------------------------------------------------------------- resolve

def test_resolve_settles_a_proposed_record(env: McpEnv, run_git: GitRunner) -> None:
    root, cfg, runtime = env
    rel = _record(env, status="proposed")
    res = tool_decision_resolve(
        cfg, runtime, path=rel, status="accepted",
        decision="Use Bases for four views.", consequences="Fewer dashboards.",
        date="2026-10-12",
    )
    assert res.committed
    text = (root / rel).read_text(encoding="utf-8")
    note = parse_decision(rel, text)
    assert (note.status, note.date, note.decided) == ("accepted", "2026-10-11", "2026-10-12")
    sections = parse_sections(text)
    assert (sections.context, sections.decision, sections.consequences) == (
        "Views were scattered.", "Use Bases for four views.", "Fewer dashboards.",
    )
    assert run_git(root, "log", "-1", "--format=%s") == (
        f"mcp(agent-a): resolve decision {rel} accepted"
    )


def test_resolve_refuses_a_settled_record(env: McpEnv) -> None:
    root, cfg, runtime = env
    rel = _record(env)
    before = (root / rel).read_bytes()
    with pytest.raises(ToolError, match="only a proposed decision"):
        tool_decision_resolve(cfg, runtime, path=rel, status="rejected")
    assert (root / rel).read_bytes() == before


def test_resolve_rolls_back_on_commit_failure(env: McpEnv, run_git: GitRunner) -> None:
    root, cfg, runtime = env
    rel = _record(env, status="proposed")
    before = (root / rel).read_bytes()
    _fail_commits(root)
    with pytest.raises(ToolError, match="nothing was written"):
        tool_decision_resolve(cfg, runtime, path=rel, status="accepted", date="2026-10-12")
    assert (root / rel).read_bytes() == before
    _assert_clean(root, run_git)
    (root / ".git" / "hooks" / "pre-commit").unlink()
    assert tool_decision_resolve(cfg, runtime, path=rel, status="accepted").committed


_HAND_FM = "topics: [obsidian]\nrelations:\n  - rel: related_to\n    target: people/anna\n"
_PREAMBLE = "Preamble written by a human in Obsidian."
_EXTRA_LINK = "- Meeting: [[knowledge/meetings/2026/2026-10-10-sync]]"


def _hand_edit(root: Path, rel: str, run_git: GitRunner) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    text = text.replace("type: decision\n", f"type: decision\n{_HAND_FM}")
    text = text.replace("## Context", f"{_PREAMBLE}\n\n## Context")
    path.write_text(text.rstrip("\n") + f"\n{_EXTRA_LINK}\n", encoding="utf-8")
    run_git(root, "commit", "-qam", "human edit")


@pytest.mark.parametrize("op", ["resolve", "supersede"])
def test_rewrites_keep_hand_added_content(env: McpEnv, run_git: GitRunner, op: str) -> None:
    root, cfg, runtime = env
    rel = _record(env, "Use Bases", status="proposed" if op == "resolve" else "accepted")
    _hand_edit(root, rel, run_git)
    if op == "resolve":
        tool_decision_resolve(cfg, runtime, path=rel, status="accepted", date="2026-10-12")
    else:
        _record(env, "Use Dataview", supersedes=1)
    text = (root / rel).read_text(encoding="utf-8")
    assert _HAND_FM in text
    assert f"# 0001 — Use Bases\n\n{_PREAMBLE}\n\n## Context" in text
    assert text.rstrip("\n").endswith(_EXTRA_LINK)
    note = parse_decision(rel, text)
    assert note.status == ("accepted" if op == "resolve" else "superseded")
    assert parse_sections(text).context == "Views were scattered."
    assert text.count("- Project: [[") == 1


def test_resolve_refuses_a_path_that_is_not_a_record(env: McpEnv) -> None:
    _root, cfg, runtime = env
    with pytest.raises(ToolError, match="not a decision record"):
        tool_decision_resolve(cfg, runtime, path=OVERVIEW, status="accepted")


# ------------------------------------------------- the protected decisions/

_GeneralVerb = Callable[["McpEnv", str], WriteResult]

_GENERAL_VERBS: dict[str, _GeneralVerb] = {
    "create": lambda e, p: tool_create_note(e[1], e[2], path=p.replace("0001", "0002"), content="x"),
    "replace": lambda e, p: tool_replace_note(e[1], e[2], path=p, content="x"),
    "append": lambda e, p: tool_append_to_note(e[1], e[2], path=p, content="x"),
    "compiled_truth": lambda e, p: tool_update_compiled_truth(e[1], e[2], path=p, content="x"),
    "append_fact": lambda e, p: tool_entity_append_fact(
        e[1], e[2], entity_path=p, text="x", source=OVERVIEW.removesuffix(".md"),
    ),
    "upsert_relation": lambda e, p: tool_entity_upsert_relation(
        e[1], e[2], entity_path=p, rel="related_to", target="projects/brain/brain",
    ),
}


@pytest.mark.parametrize("verb", sorted(_GENERAL_VERBS))
@pytest.mark.parametrize("upper", [False, True])
def test_general_verbs_refuse_decision_paths(env: McpEnv, verb: str, upper: bool) -> None:
    root, _cfg, _runtime = env
    rel = _record(env)
    before = (root / rel).read_bytes()
    target = rel.replace("/decisions/", "/Decisions/") if upper else rel
    with pytest.raises(ToolError, match="decision_record"):
        _GENERAL_VERBS[verb](env, target)
    assert (root / rel).read_bytes() == before


@pytest.mark.parametrize("folder", [DECISIONS, DECISIONS.replace("decisions", "Decisions")])
def test_create_refuses_a_file_named_decisions(env: McpEnv, folder: str) -> None:
    root, cfg, runtime = env
    with pytest.raises(ToolError, match="decision_record"):
        tool_create_note(cfg, runtime, path=folder, content="x")
    assert not (root / DECISIONS).exists()
    assert _record(env).startswith(f"{DECISIONS}/")


# ----------------------------------------------------- rollback (one commit)

def test_supersede_rolls_back_on_commit_failure(env: McpEnv, run_git: GitRunner) -> None:
    root, _cfg, _runtime = env
    old_rel = _record(env, "Use Bases")
    before = (root / old_rel).read_bytes()
    head = run_git(root, "rev-parse", "HEAD")
    _fail_commits(root)
    with pytest.raises(ToolError, match="nothing was written"):
        _record(env, "Use Dataview", supersedes=1)
    assert not (root / DECISIONS / "0002-use-dataview.md").exists()
    assert (root / old_rel).read_bytes() == before
    assert run_git(root, "rev-parse", "HEAD") == head
    _assert_clean(root, run_git)


def test_supersede_rolls_back_when_the_second_write_fails(
    env: McpEnv, run_git: GitRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _cfg, _runtime = env
    old_rel = _record(env, "Use Bases")
    before = (root / old_rel).read_bytes()
    real_write = tools_mod._atomic_write_text
    calls: list[Path] = []

    def _second_write_fails(target: Path, text: str) -> None:
        calls.append(target)
        if len(calls) == 2:
            raise ToolError("write failed")
        real_write(target, text)

    monkeypatch.setattr(tools_mod, "_atomic_write_text", _second_write_fails)
    with pytest.raises(ToolError, match="write failed"):
        _record(env, "Use Dataview", supersedes=1)
    assert len(calls) == 2
    assert not (root / DECISIONS / "0002-use-dataview.md").exists()
    assert (root / old_rel).read_bytes() == before
    _assert_clean(root, run_git)


def test_meeting_create_rolls_back_on_commit_failure(env: McpEnv, run_git: GitRunner) -> None:
    root, cfg, runtime = env
    anna = root / "knowledge/people/anna.md"
    before = anna.read_bytes()
    _fail_commits(root)
    with pytest.raises(ToolError, match="nothing was written"):
        tool_meeting_create(
            cfg, runtime, date="2026-10-11", title="Sync", attendees=["people/anna"],
        )
    assert not (root / "knowledge/meetings/2026/2026-10-11-sync.md").exists()
    assert anna.read_bytes() == before
    _assert_clean(root, run_git)
