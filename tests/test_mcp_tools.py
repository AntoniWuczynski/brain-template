"""Two write-path guarantees, both about what a tool refuses to clobber.

``vault_update_compiled_truth``: the marker-scoped write tool. The dream
pass's compiled-truth job used to send whole entity notes through
``vault_replace_note``, with "only the text between the markers changes"
enforced by prose in a skill file. These tests are the mechanical version of
that promise: the server splices between the fence and refuses everything
else.

Pull-before-write (AUD-122): every audited write first absorbs the commits
the owner's Obsidian clone pushed to the same remote. Those tests drive two
real clones of one bare repo through the write tools.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import CfgFactory, GitRunner, McpEnv

from ingest_lib.dream import COMPILED_TRUTH_END, COMPILED_TRUTH_START
from mcp_server.audit import AuditLog
from mcp_server.errors import ToolError
from mcp_server.identity import AGENT_VAR
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import (
    tool_append_to_note,
    tool_create_note,
    tool_update_compiled_truth,
)

pytestmark = pytest.mark.usefixtures("roomy_rate_buckets")

_NOTE = (
    "---\n"
    "title: Anna\n"
    "type: person\n"
    "relations:\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "---\n"
    "\n"
    f"{COMPILED_TRUTH_START}\n"
    "Anna works at Acme.\n"
    f"{COMPILED_TRUTH_END}\n"
    "\n"
    "## Overview\n"
    "\n"
    "Hand-written prose that must survive.\n"
    "\n"
    "## Log\n"
    "\n"
    "- 2026-06-01 — joined the platform team ([[knowledge/x]])\n"
)


# The server stamps these on every write (AGENTS.md); they are the only
# frontmatter lines this tool is allowed to add.
_PROVENANCE_KEYS = ("last_written_by:", "written_via:")


def _write(root: Path, rel: str, text: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _without_provenance(text: str) -> str:
    """``text`` with the server-stamped provenance lines dropped, so a test
    can compare every OTHER byte of the note."""
    return "".join(
        line for line in text.splitlines(keepends=True)
        if not line.startswith(_PROVENANCE_KEYS)
    )


def test_update_compiled_truth_replaces_only_the_fenced_block(
    mcp_env: McpEnv,
) -> None:
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    tool_update_compiled_truth(
        cfg, runtime,
        path="knowledge/people/anna.md",
        content="Anna left Acme in June 2026 and has no current employer.",
    )

    after = note.read_text(encoding="utf-8")
    assert _without_provenance(after) == _NOTE.replace(
        "Anna works at Acme.",
        "Anna left Acme in June 2026 and has no current employer.",
    )
    # Everything outside the fence, byte for byte.
    assert "  - rel: works_at" in after
    assert "Hand-written prose that must survive." in after
    assert "- 2026-06-01 — joined the platform team ([[knowledge/x]])" in after


def test_update_compiled_truth_stamps_provenance(mcp_env: McpEnv) -> None:
    """N6: AGENTS.md says the server stamps every note it writes. The
    stamped values come from the agent, not the clock, so two runs by the
    same agent leave the frontmatter byte-identical — which is what keeps
    ``dream._only_compiled_truth_changed`` seeing a fence-only edit."""
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="First truth.",
    )
    first = note.read_text(encoding="utf-8")
    assert "last_written_by: 'agent:agent-a'" in first
    assert "written_via: mcp" in first

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Second truth.",
    )
    second = note.read_text(encoding="utf-8")
    assert second.replace("Second truth.", "First truth.") == first


def test_update_compiled_truth_bootstraps_the_first_fence_below_the_frontmatter(
    mcp_env: McpEnv,
) -> None:
    """N2: the note has no fence yet. The server puts the first one directly
    under the frontmatter — never appended, because an entity note's last
    section is ``## Log`` and a fence inside it is read back as Log evidence
    by the very pass that wrote it."""
    root, cfg, runtime = mcp_env
    plain = _NOTE.replace(
        f"{COMPILED_TRUTH_START}\nAnna works at Acme.\n{COMPILED_TRUTH_END}\n\n", ""
    )
    note = _write(root, "knowledge/people/anna.md", plain)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Anna works at Acme.",
    )

    after = _without_provenance(note.read_text(encoding="utf-8"))
    assert after == _NOTE
    assert after.index(COMPILED_TRUTH_END) < after.index("## Overview")
    assert after.index(COMPILED_TRUTH_START) < after.index("## Log")


def test_update_compiled_truth_bootstrap_ignores_a_documented_fence(
    mcp_env: McpEnv,
) -> None:
    """N7: a note quoting the markers inside a ``` block has no fence of its
    own. Counting the quoted pair read the note as fenced and overwrote the
    example."""
    root, cfg, runtime = mcp_env
    documented = (
        "---\ntitle: Anna\ntype: person\n---\n\n"
        "## Overview\n\nThe block looks like this:\n\n"
        f"```markdown\n{COMPILED_TRUTH_START}\n...\n{COMPILED_TRUTH_END}\n```\n"
    )
    note = _write(root, "knowledge/people/anna.md", documented)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Anna works at Acme.",
    )

    after = _without_provenance(note.read_text(encoding="utf-8"))
    assert after == documented.replace(
        "---\n\n## Overview",
        f"---\n\n{COMPILED_TRUTH_START}\nAnna works at Acme.\n"
        f"{COMPILED_TRUTH_END}\n\n## Overview",
    )
    assert f"```markdown\n{COMPILED_TRUTH_START}\n...\n{COMPILED_TRUTH_END}\n```" in after


def test_update_compiled_truth_refuses_content_carrying_a_marker(
    mcp_env: McpEnv,
) -> None:
    """N1 case A: a payload that closes the fence itself escapes it. The
    splice was verbatim, so the text landed AFTER the END marker and the
    note was left with two ENDs — unwritable by this tool for ever."""
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    with pytest.raises(ToolError, match="must not contain the fence markers"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/anna.md",
            content=f"evil\n{COMPILED_TRUTH_END}\nESCAPED PAST THE FENCE\n",
        )
    assert note.read_text(encoding="utf-8") == _NOTE


def test_update_compiled_truth_refuses_the_skills_own_example_block(
    mcp_env: McpEnv,
) -> None:
    """N1 case L, the likeliest real trigger: the model copies the skill's
    step-6 example and sends the whole block, markers included."""
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    with pytest.raises(ToolError, match="must not contain the fence markers"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/anna.md",
            content=(
                f"{COMPILED_TRUTH_START}\n"
                "Anna works at Acme.\n"
                f"{COMPILED_TRUTH_END}"
            ),
        )
    assert note.read_text(encoding="utf-8") == _NOTE


def test_update_compiled_truth_refuses_a_broken_fence(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    half = _NOTE.replace(f"{COMPILED_TRUTH_END}\n", "")
    note = _write(root, "knowledge/people/anna.md", half)

    with pytest.raises(ToolError, match="ambiguous compiled-truth fence"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/anna.md", content="anything",
        )
    assert note.read_text(encoding="utf-8") == half


def test_update_compiled_truth_refuses_the_memory_areas(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    fact = _write(root, "knowledge/assistant/inbox/x.md", _NOTE)

    with pytest.raises(ToolError, match="memory lifecycle"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/assistant/inbox/x.md", content="anything",
        )
    assert fact.read_text(encoding="utf-8") == _NOTE


def test_update_compiled_truth_refuses_a_missing_note(mcp_env: McpEnv) -> None:
    _root, cfg, runtime = mcp_env
    with pytest.raises(ToolError, match="does not exist"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/nobody.md", content="anything",
        )


def test_update_compiled_truth_audits_the_write(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    _write(root, "knowledge/people/anna.md", _NOTE)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Fresh truth.",
    )

    audit = (root / "logs" / "mcp-audit.jsonl").read_text(encoding="utf-8")
    assert "vault_update_compiled_truth" in audit


# ---------------------------------------------- pull before write (AUD-122)
#
# After the migration the server's clone and the owner's Obsidian clone both
# push to one remote. Every audited write fetches and rebases first, so the
# server absorbs hand edits instead of drifting permanently behind them.


@pytest.fixture()
def as_agent_a() -> Iterator[None]:
    """Pin the request identity the way the auth middleware would."""
    token = AGENT_VAR.set("agent-a")
    try:
        yield
    finally:
        AGENT_VAR.reset(token)


def _identity(repo: Path, run_git: GitRunner) -> None:
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "test")


def _shared_remote(tmp_path: Path, run_git: GitRunner) -> tuple[Path, Path]:
    """``(server, obsidian)`` — two clones of one bare remote, seeded with
    a knowledge note so both sides can edit the same file."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    seed = tmp_path / "seed"
    (seed / "knowledge" / "notes").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    _identity(seed, run_git)
    (seed / "knowledge" / "notes" / "shared.md").write_text("seed\n", encoding="utf-8")
    run_git(seed, "add", "-A")
    run_git(seed, "commit", "-q", "-m", "seed")
    run_git(seed, "remote", "add", "origin", str(bare))
    run_git(seed, "push", "-q", "origin", "main")

    clones: list[Path] = []
    for name in ("server", "obsidian"):
        clone = tmp_path / name
        subprocess.run(["git", "clone", "-q", str(bare), str(clone)], check=True)
        _identity(clone, run_git)
        clones.append(clone)
    return clones[0], clones[1]


def _runtime_with_remote(root: Path) -> Runtime:
    """A Runtime whose push worker is configured (so pull-before-write is
    on) but whose background thread is stopped.

    These tests are about the pull each WRITE does; the worker's own
    startup push would race their git assertions. Build it while the two
    clones are still in sync — its startup push is then a no-op — and stop
    it before the test diverges them, after which ``request_push`` only
    sets a flag. ``push_enabled`` stays True, which is what gates the pull.
    """
    audit = AuditLog(root)
    worker = PushWorker(
        root, remote="origin", branch="main", enabled=True,
        retry_schedule=(3600.0,),
    )
    worker.stop(flush_seconds=5.0)
    return Runtime(
        audit=audit,
        push_worker=worker,
        refresher=IndexRefresher(root, audit=audit, enabled=False),
    )


def _synced_trio(tmp_path: Path, run_git: GitRunner) -> tuple[Path, Path, Runtime]:
    """``(server, obsidian, runtime)`` with both clones still in sync."""
    server, obsidian = _shared_remote(tmp_path, run_git)
    return server, obsidian, _runtime_with_remote(server)


def _hand_edit(obsidian: Path, run_git: GitRunner, *, rel: str, text: str, msg: str) -> None:
    """The owner edits a note in Obsidian, commits and pushes it."""
    target = obsidian / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    run_git(obsidian, "add", "-A")
    run_git(obsidian, "commit", "-q", "-m", msg)
    run_git(obsidian, "push", "-q", "origin", "main")


def _subjects(repo: Path, run_git: GitRunner) -> list[str]:
    out = run_git(repo, "log", "--format=%s")
    return out.splitlines() if out else []


def _audit_rows(root: Path) -> list[dict[str, str | None]]:
    path = root / "logs" / "mcp-audit.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, str | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.mark.usefixtures("as_agent_a")
def test_write_fast_forwards_over_a_hand_edit(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory
) -> None:
    server, obsidian, runtime = _synced_trio(tmp_path, run_git)
    _hand_edit(
        obsidian, run_git,
        rel="knowledge/notes/hand.md", text="typed by hand\n", msg="hand edit",
    )

    result = tool_create_note(
        make_cfg(server), runtime,
        path="knowledge/notes/agent.md", content="# Agent note\n",
    )

    assert result.committed is True
    # The hand edit is present, and the agent's commit sits on top of it.
    assert (server / "knowledge" / "notes" / "hand.md").exists()
    subjects = _subjects(server, run_git)
    assert "create note knowledge/notes/agent.md" in subjects[0]
    assert subjects[1] == "hand edit"
    rebased = [r for r in _audit_rows(server) if r["outcome"] == "rebased"]
    assert len(rebased) == 1
    assert rebased[0]["detail"] == "pull-before-write absorbed 1 commit(s) by fast-forward"


@pytest.mark.usefixtures("as_agent_a")
def test_write_rebases_local_commits_over_a_hand_edit(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory
) -> None:
    server, obsidian, runtime = _synced_trio(tmp_path, run_git)
    # An earlier write that has not been pushed yet.
    (server / "knowledge" / "notes" / "earlier.md").write_text("earlier\n", encoding="utf-8")
    run_git(server, "add", "-A")
    run_git(server, "commit", "-q", "-m", "earlier write")
    _hand_edit(
        obsidian, run_git,
        rel="knowledge/notes/hand.md", text="typed by hand\n", msg="hand edit",
    )

    result = tool_create_note(
        make_cfg(server), runtime,
        path="knowledge/notes/agent.md", content="# Agent note\n",
    )

    assert result.committed is True
    subjects = _subjects(server, run_git)
    assert "create note knowledge/notes/agent.md" in subjects[0]
    assert subjects[1] == "earlier write"
    assert subjects[2] == "hand edit"
    rebased = [r for r in _audit_rows(server) if r["outcome"] == "rebased"]
    assert rebased[0]["detail"] == "pull-before-write absorbed 1 commit(s) by rebase"


@pytest.mark.usefixtures("as_agent_a")
def test_write_is_refused_when_the_hand_edit_conflicts(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory
) -> None:
    """IMP-029: refuse, name the path, change nothing on either side."""
    server, obsidian, runtime = _synced_trio(tmp_path, run_git)
    _hand_edit(
        obsidian, run_git,
        rel="knowledge/notes/shared.md", text="edited by hand\n", msg="hand edit",
    )
    shared = server / "knowledge" / "notes" / "shared.md"
    shared.write_text("edited by the agent\n", encoding="utf-8")
    run_git(server, "add", "-A")
    run_git(server, "commit", "-q", "-m", "agent edit")
    head_before = run_git(server, "rev-parse", "HEAD")

    with pytest.raises(ToolError, match="knowledge/notes/shared.md"):
        tool_append_to_note(
            make_cfg(server), runtime,
            path="knowledge/notes/shared.md", content="more\n",
        )

    assert run_git(server, "rev-parse", "HEAD") == head_before
    assert run_git(server, "status", "--porcelain", "--untracked-files=no") == ""
    assert shared.read_text(encoding="utf-8") == "edited by the agent\n"
    refusals = [
        r for r in _audit_rows(server)
        if r["outcome"] is not None and str(r["outcome"]).startswith("refused:")
    ]
    assert len(refusals) == 1
    assert "conflicting edits" in str(refusals[0]["outcome"])


@pytest.mark.usefixtures("as_agent_a")
def test_write_succeeds_while_the_remote_is_unreachable(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory
) -> None:
    """The laptop case: no network must not cost the agent its write."""
    server, _obsidian, runtime = _synced_trio(tmp_path, run_git)
    run_git(server, "remote", "set-url", "origin", str(tmp_path / "gone.git"))

    result = tool_create_note(
        make_cfg(server), runtime,
        path="knowledge/notes/offline.md", content="# Offline\n",
    )

    assert result.committed is True
    assert result.commit_sha is not None
    assert "create note knowledge/notes/offline.md" in _subjects(server, run_git)[0]
    assert run_git(server, "status", "--porcelain", "--untracked-files=no") == ""
    assert [r for r in _audit_rows(server) if r["outcome"] == "rebased"] == []


@pytest.mark.usefixtures("as_agent_a")
def test_write_proceeds_when_the_tree_is_dirty_and_reports_the_skipped_pull(
    tmp_path: Path, run_git: GitRunner, make_cfg: CfgFactory
) -> None:
    """A dirty tracked file is a hand edit in flight from OUTSIDE this
    process — Obsidian on the Mac, another session — and that is the working
    clone's normal state before the migration. A rebase cannot run over it,
    so the pull is skipped and said so; the write still commits its own
    paths only, and the hand edit is left exactly as it was."""
    server, _obsidian, runtime = _synced_trio(tmp_path, run_git)
    shared = server / "knowledge" / "notes" / "shared.md"
    shared.write_text("half\n", encoding="utf-8")

    result = tool_create_note(
        make_cfg(server), runtime,
        path="knowledge/notes/agent.md", content="# Agent note\n",
    )

    assert result.committed
    assert (server / "knowledge" / "notes" / "agent.md").exists()
    assert shared.read_text(encoding="utf-8") == "half\n"          # hand edit untouched
    assert "shared.md" in run_git(server, "status", "--porcelain")  # and still uncommitted
    outcomes = [row["outcome"] for row in _audit_rows(server)]
    assert "pull-skipped" in outcomes
    assert outcomes[-1] == "ok"
    assert runtime.push_worker.status().get("sync_error")

