"""Dream-pass SKILL.md and dream.sh contract checks (split out of
test_dream.py, AUD-121): the skill copies stay in sync, name the closed
type vocabulary, and describe the tools/scratch paths dream.sh actually
grants."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ingest_lib.dream import COMPILED_TRUTH_END, COMPILED_TRUTH_START

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL_COPIES = (
    _REPO_ROOT / ".claude/skills/dream-pass/SKILL.md",
    _REPO_ROOT / "_template/.claude/skills/dream-pass/SKILL.md",
    _REPO_ROOT / ".agents/skills/dream-pass/SKILL.md",
)


def _skill_texts() -> list[tuple[Path, str]]:
    return [(p, p.read_text(encoding="utf-8")) for p in _SKILL_COPIES if p.is_file()]


def test_skill_pins_every_dream_note_to_the_closed_vocabulary_type() -> None:
    """AGENTS.md's closed type vocabulary has no connection/report/question
    member, so the skill must name the one value it may write."""
    for path, text in _skill_texts():
        assert "`type:` is always `digest`" in text, path
        assert "type: digest" in text, path
        for invented in ("type: connection", "type: report", "type: question", "type: log"):
            assert invented not in text, f"{path}: {invented}"


def test_skill_gate_step_records_no_pending_marker() -> None:
    """Step 1 runs the gate dry: the skill has no way to clear a pending
    marker on its early exits, so it must never stamp one."""
    for path, text in _skill_texts():
        assert "scripts/dream_gate.py --dry-run" in text, path
        assert "1. **Gate.** Run `uv run --no-sync python scripts/dream_gate.py`" not in text, path


def test_skill_copies_agree_outside_the_sync_note() -> None:
    texts = _skill_texts()
    if len(texts) < 2:
        # Private-repo check. push_to_upstream.sh publishes exactly one copy
        # (from _template/.claude/skills/); .agents/ and _template/ itself are
        # never synced, so there is nothing to compare on the template branch.
        pytest.skip(
            "private-repo check: only one dream-pass SKILL.md copy is present "
            f"({[str(p) for p, _ in texts]}); the other copies live in "
            "_template/ and .agents/, which do not sync to the template"
        )
    stripped = [
        [ln for ln in text.splitlines() if "skills/dream-pass/SKILL.md` in the private" not in ln]
        for _path, text in texts
    ]
    assert all(block == stripped[0] for block in stripped[1:])


def test_dream_sh_caps_the_runner_wall_clock() -> None:
    """--max-turns bounds turns, not time; every launched runner goes
    through the timeout wrapper, which escalates to SIGKILL."""
    script = (_REPO_ROOT / "scripts/dream.sh").read_text(encoding="utf-8")
    assert "--kill-after=60s" in script
    assert '"$TIMEOUT_BIN" --kill-after=60s "$TIMEOUT_SECS" "$@"' in script
    assert "run_capped claude -p" in script
    assert "run_capped codex exec" in script
    subprocess.run(["bash", "-n", str(_REPO_ROOT / "scripts/dream.sh")], check=True)


_SCRATCH_REL = "~/.cache/brain-dream"
_SURVIVORS = f"{_SCRATCH_REL}/survivors.json"


def test_dream_sh_grants_only_the_tools_the_skill_uses() -> None:
    """M6: `mcp__brain__*` handed the nightly session every write tool the
    server has — entity_upsert_relation and profile_update included, which
    move the graph the pass is only allowed to PROPOSE changes to."""
    script = (_REPO_ROOT / "scripts/dream.sh").read_text(encoding="utf-8")
    assert '"mcp__brain__*"' not in script
    for granted in (
        "mcp__brain__vault_read",
        "mcp__brain__vault_search",
        "mcp__brain__vault_related",
        "mcp__brain__vault_create_note",
        # steps 5, 8 and 9 replace a digest, an entity note and questions.md
        "mcp__brain__vault_replace_note",
        "mcp__brain__vault_update_compiled_truth",
    ):
        assert granted in script, granted
    for withheld in (
        "mcp__brain__entity_upsert_relation",
        "mcp__brain__profile_update",
        "mcp__brain__entity_append_fact",
        "mcp__brain__vault_drop_inbox_file",
        "mcp__brain__vault_append_to_note",
    ):
        assert withheld not in script, withheld


def test_dream_sh_grants_the_write_step_7_needs() -> None:
    """N3: the skill's step 7 writes survivors.json and feeds it to
    `dream_gate.py --propose`. With no Write in --allowedTools the scheduled
    pass could never produce a contradiction proposal at all."""
    script = (_REPO_ROOT / "scripts/dream.sh").read_text(encoding="utf-8")
    assert f'"Write({_SCRATCH_REL}/**)"' in script
    assert '"Write(/$SCRATCH/**)"' in script
    assert f'SCRATCH="$HOME/{_SCRATCH_REL.removeprefix("~/")}"' in script
    assert 'mkdir -p "$SCRATCH"' in script
    for path, text in _skill_texts():
        assert _SURVIVORS in text, path
        assert "--propose survivors.json" not in text, path


def test_skill_bootstraps_a_missing_fence_through_the_tool() -> None:
    """N2: the old bootstrap appended an empty fence, which on an entity
    note lands inside the ## Log — where the next pass reads it back as Log
    evidence. The server places the first fence now."""
    for path, text in _skill_texts():
        assert "mcp__brain__vault_append_to_note" not in text.split("Do not add a fence")[0], path
        assert "the server creates the first one, directly below the" in text, path


def test_skill_names_the_compiled_truth_markers_the_packet_detects() -> None:
    """dream.py decides marker_state from these exact strings; a skill that
    writes a different fence would be invisible to the next packet."""
    for path, text in _skill_texts():
        assert COMPILED_TRUTH_START in text, path
        assert COMPILED_TRUTH_END in text, path
        assert "compiled_truth_blocked" in text, path


def test_skill_keeps_contradiction_findings_propose_only() -> None:
    for path, text in _skill_texts():
        assert "**This job never edits an entity note, and never writes the proposal" in text, path
        assert "scripts/dream_gate.py --propose" in text, path
        assert "`proposal_fact` and `promote_relations` go in unaltered" in text, path
        assert "approved: false" in text, path
        assert "Only a human approves" in text, path


def test_skill_writes_proposals_through_the_deterministic_proposer() -> None:
    """M5: the frontmatter contract has one owner. The skill must not tell
    the agent to hand-author a memory-fact note."""
    for path, text in _skill_texts():
        assert "scripts/ingest_lib/propose.py" in text, path
        assert "author: script:dream-contradiction" in text, path
        assert "type: memory_fact\n   generated_by: dream-pass" not in text, path


def test_skill_never_promotes_an_unapproved_inbox_proposal() -> None:
    """refute-propose M2: the pass's own output must not be read back as
    fact, and approval stays a human's decision that consolidate executes."""
    for path, text in _skill_texts():
        assert "**Unapproved proposals are not facts.**" in text, path
        assert "never set `approved: true`" in text, path
        assert "`scripts/consolidate.py` is what executes it" in text, path


def test_skill_writes_compiled_truth_only_through_the_marker_scoped_tool() -> None:
    """M6: the 'only between the markers' promise is a tool refusal now,
    so the skill must name that tool and rule out the general replace."""
    for path, text in _skill_texts():
        assert "mcp__brain__vault_update_compiled_truth" in text, path
        assert "**never**\n   `mcp__brain__vault_replace_note`" in text, path
        assert "Never write compiled truth\n  with `mcp__brain__vault_replace_note`." in text, path
