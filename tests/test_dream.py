"""Dream-pass gate: state IO, pending marker, gate arithmetic, packet
determinism. All vaults are tmp_path fixtures with a real git repo so the
gate's git arithmetic runs against genuine history; commit dates are
pinned via GIT_AUTHOR_DATE/GIT_COMMITTER_DATE for determinism."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.consolidate import consolidate
from ingest_lib.dream import (
    AdjudicatedFinding,
    COMPILED_TRUTH_END,
    COMPILED_TRUTH_START,
    CONTRADICTION_FLAG_PREFIX,
    ContradictionCandidate,
    DreamState,
    build_packet,
    evaluate_gate,
    load_pending_since,
    load_state,
    mark_done,
    marker_state,
    parse_adjudicated,
    record_pending,
    write_proposals,
)
from ingest_lib.dream import _restated_facts  # private: the emission cap is its own
from ingest_lib.hashing import sha256_of
from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import parse_relations
from ingest_lib.propose import propose_fact
from ingest_lib.relations import Relation
from ingest_lib.metadata import IndexRecord, append_record

_LOG = logging.getLogger("test")
AS_OF = datetime(2026, 6, 12, 5, 0, tzinfo=UTC)


def _git(root: Path, *args: str, when: str | None = None) -> str:
    env = dict(os.environ)
    if when is not None:
        env["GIT_AUTHOR_DATE"] = when
        env["GIT_COMMITTER_DATE"] = when
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=True, env=env,
    )
    return result.stdout


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".keep").write_text("", encoding="utf-8")
    _commit(paths, "init", when="2026-05-01T12:00:00+00:00")
    return paths


def _commit(paths: VaultPaths, msg: str, *, when: str) -> str:
    _git(paths.root, "add", "-A")
    # --allow-empty: some tests commit purely to mint a base sha
    _git(paths.root, "commit", "-q", "--allow-empty", "-m", msg, when=when)
    return _git(paths.root, "rev-parse", "HEAD").strip()


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    target = paths.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_load_state_missing_returns_none(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    assert load_state(paths) is None


def test_load_state_malformed_returns_none(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (paths.metadata / "dream.json").write_text("not json", encoding="utf-8")
    assert load_state(paths) is None
    (paths.metadata / "dream.json").write_text('{"last_run": 3}', encoding="utf-8")
    assert load_state(paths) is None


def test_mark_done_writes_state_and_clears_pending(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    record_pending(paths, now=AS_OF)
    assert (paths.metadata / "dream.pending").is_file()

    state = mark_done(paths, now=AS_OF, logger=_LOG)

    head = _git(paths.root, "rev-parse", "HEAD").strip()
    assert state == DreamState(last_run="2026-06-12T05:00:00Z", last_commit=head)
    assert load_state(paths) == state
    assert not (paths.metadata / "dream.pending").exists()


def test_record_pending_first_detection_wins(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    record_pending(paths, now=AS_OF)
    later = datetime(2026, 6, 14, 5, 0, tzinfo=UTC)
    record_pending(paths, now=later)  # must NOT overwrite the first stamp
    since = load_pending_since(paths)
    assert since == AS_OF


def test_load_pending_since_absent_or_malformed(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    assert load_pending_since(paths) is None
    (paths.metadata / "dream.pending").write_text("garbage", encoding="utf-8")
    assert load_pending_since(paths) is None


def _note(i: int) -> str:
    return f"knowledge/notes/n{i}.md"


def _add_notes(paths: VaultPaths, count: int, *, when: str, start: int = 0) -> str:
    for i in range(start, start + count):
        _write(paths, _note(i), f"# n{i}\nbody\n")
    return _commit(paths, f"add {count} notes", when=when)


def _add_source_record(paths: VaultPaths, rel: str, created_at: str) -> None:
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("source", encoding="utf-8")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel,
        source_hash=sha256_of(raw),
        size_bytes=6,
        extension=".txt",
        extractor="text",
        status="processed",
        raw_path=f"archive/raw/{rel}",
        processed_path=None,
        index_note_path=None,
        created_at=created_at,
    ))


def test_gate_first_run_nothing_recent_skips(tmp_path: Path) -> None:
    paths = _vault(tmp_path)  # only the 2026-05-01 init commit, > 7 days old
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    assert not verdict.should_dream
    assert verdict.days_since_last is None
    assert verdict.changed_notes == ()


def test_gate_first_run_with_recent_change_dreams(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")  # inside the 7-day window
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    assert verdict.should_dream
    assert verdict.changed_notes == (_note(0),)


def test_gate_below_threshold_skips(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 4, when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert not verdict.should_dream
    assert len(verdict.changed_notes) == 4
    assert verdict.days_since_last == 1


def test_gate_at_threshold_dreams(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 5, when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert verdict.should_dream
    assert verdict.base_commit == base


def test_gate_stale_override_dreams_on_one_change(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-01T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-01T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 1, when="2026-06-02T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5, stale_days=7)
    assert verdict.should_dream
    assert "stale" in verdict.reason


def test_gate_counts_new_index_sources(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    for i in range(5):
        _add_source_record(paths, f"inbox/s{i}.txt", "2026-06-11T09:00:00Z")
    _add_source_record(paths, "inbox/old.txt", "2026-06-01T09:00:00Z")  # before last_run
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert verdict.should_dream
    assert len(verdict.new_sources) == 5
    assert "inbox/old.txt" not in verdict.new_sources


def test_gate_missing_base_commit_falls_back(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": "0" * 40, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 1, when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)  # must not raise
    assert verdict.changed_notes == (_note(0),)


def test_gate_concept_notes_only_does_not_trip_threshold(tmp_path: Path) -> None:
    """A derived-notes refresh (concept notes rewritten wholesale) is not
    'new information' — it must not count towards the gate's threshold."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    for i in range(6):  # over the default threshold of 5
        _write(paths, f"knowledge/concepts/c{i}.md",
               f"---\ntitle: c{i}\ntype: concept\n---\n# c{i}\n")
    _commit(paths, "refresh concepts", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert not verdict.should_dream
    assert verdict.changed_notes == ()


def test_gate_mixed_commit_counts_only_hand_written_notes(tmp_path: Path) -> None:
    """A commit mixing generated notes (concepts, index dashboards, a
    dream-pass connection note) with hand/MCP-written notes counts and
    lists only the latter."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    for i in range(4):
        _write(paths, f"knowledge/concepts/c{i}.md",
               f"---\ntitle: c{i}\ntype: concept\n---\n# c{i}\n")
    _write(paths, "knowledge/index/entities/people.md",
           "---\ntitle: People\ntype: dashboard\n---\n# People\n")
    _write(paths, "knowledge/notes/dreams/connections/x--y.md",
           "---\ntitle: x\ntype: connection\ngenerated_by: dream-pass\n---\n# x\n")
    _write(paths, "knowledge/people/anna.md", "---\ntitle: Anna\ntype: person\n---\n# Anna\n")
    _write(paths, "knowledge/projects/brain/log/2026-06-11.md", "# log\n")
    _commit(paths, "mixed", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert not verdict.should_dream  # only 2 real notes, below threshold
    assert verdict.changed_notes == (
        "knowledge/people/anna.md",
        "knowledge/projects/brain/log/2026-06-11.md",
    )


def _write_connections(paths: VaultPaths, lines: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(ln, sort_keys=True) for ln in lines) + "\n"
    (paths.metadata / "connections.jsonl").write_text(text, encoding="utf-8")


def test_packet_candidate_pairs_semantic_without_cooccurrence(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")
    _write_connections(paths, [
        {"a": "alpha", "b": "beta", "kind": "semantic", "weight": 0.9, "sources": []},
        {"a": "alpha", "b": "beta", "kind": "cooccurrence", "weight": 2.0, "sources": ["x"]},
        {"a": "gamma", "b": "delta", "kind": "semantic", "weight": 0.7, "sources": []},
        {"a": "gamma", "b": "epsilon", "kind": "semantic", "weight": 0.8, "sources": []},
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=10, logger=_LOG)
    # (alpha, beta) is already linked by cooccurrence -> excluded;
    # the rest ranked by weight descending.
    assert packet["candidate_pairs"] == [
        {"a": "gamma", "b": "epsilon", "weight": 0.8},
        {"a": "gamma", "b": "delta", "weight": 0.7},
    ]


def test_packet_top_k_caps_pairs(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")
    _write_connections(paths, [
        {"a": f"c{i}", "b": f"d{i}", "kind": "semantic", "weight": 0.5 + i / 100, "sources": []}
        for i in range(15)
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=3, logger=_LOG)
    assert len(packet["candidate_pairs"]) == 3
    assert packet["candidate_pairs"][0]["a"] == "c14"


def test_packet_active_entities_and_dream_inventory(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", "# anna\n")
    _write(paths, "knowledge/projects/brain/log/2026-06-11.md", "# log\n")
    _write(paths, "knowledge/notes/dreams/digests/brain-week.md", "# digest\n")
    _commit(paths, "entities", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=10, logger=_LOG)
    assert packet["active_entities"] == [
        "knowledge/people/anna",
        "knowledge/projects/brain",
    ]
    assert packet["existing_dream_notes"] == [
        "knowledge/notes/dreams/digests/brain-week.md",
    ]


def test_packet_changed_notes_excludes_generated(tmp_path: Path) -> None:
    """The packet's worklist mirrors the gate's filtered changed_notes/
    active_entities — generated notes never reach the LLM session either."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _write(paths, "knowledge/concepts/c0.md", "---\ntitle: c0\ntype: concept\n---\n# c0\n")
    _write(paths, "knowledge/people/anna.md", "---\ntitle: Anna\ntype: person\n---\n# Anna\n")
    _commit(paths, "mixed", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    assert verdict.should_dream
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["changed_notes"] == ["knowledge/people/anna.md"]
    assert packet["active_entities"] == ["knowledge/people/anna"]


def test_packet_deterministic(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 2, when="2026-06-10T12:00:00+00:00")
    _write_connections(paths, [
        {"a": "x", "b": "y", "kind": "semantic", "weight": 0.6, "sources": []},
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    one = json.dumps(build_packet(paths, verdict=verdict, top_k=5, logger=_LOG), sort_keys=True)
    two = json.dumps(build_packet(paths, verdict=verdict, top_k=5, logger=_LOG), sort_keys=True)
    assert one == two


def test_packet_drops_pairs_that_already_have_a_connection_note(tmp_path: Path) -> None:
    """A pair the dream already explained must not keep occupying a top_k
    slot — in either file-name order, since older notes were written as
    <b>--<a>."""
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")
    _write(paths, "knowledge/notes/dreams/connections/alpha--beta.md", "# done\n")
    _write(paths, "knowledge/notes/dreams/connections/delta--gamma.md", "# done\n")
    _write_connections(paths, [
        {"a": "alpha", "b": "beta", "kind": "semantic", "weight": 0.9, "sources": []},
        {"a": "gamma", "b": "delta", "kind": "semantic", "weight": 0.8, "sources": []},
        {"a": "gamma", "b": "epsilon", "kind": "semantic", "weight": 0.7, "sources": []},
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=10, logger=_LOG)
    assert packet["candidate_pairs"] == [
        {"a": "gamma", "b": "epsilon", "weight": 0.7},
    ]


def test_packet_fills_top_k_after_dropping_written_pairs(tmp_path: Path) -> None:
    """Dedup happens before the top_k truncation, so handled pairs free
    their slots for the next-ranked unexplained ones."""
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")
    for i in range(4, 7):
        _write(paths, f"knowledge/notes/dreams/connections/c{i}--d{i}.md", "# done\n")
    _write_connections(paths, [
        {"a": f"c{i}", "b": f"d{i}", "kind": "semantic", "weight": 0.5 + i / 100, "sources": []}
        for i in range(7)
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=3, logger=_LOG)
    assert [p["a"] for p in packet["candidate_pairs"]] == ["c3", "c2", "c1"]


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
    assert len(texts) >= 2
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


# --------------------------------------------------------------------------
# compiled truth
# --------------------------------------------------------------------------

def _entity(relations: str = "", log: str = "", *, marker: str = "") -> str:
    """An entity note: frontmatter (optionally with relations), an optional
    compiled-truth fence, a hand-written body, then the ## Log."""
    head = "---\ntitle: Anna\ntype: person\n"
    if relations:
        head += relations
    head += "---\n\n"
    if marker:
        head += marker + "\n\n"
    head += "Hand-written prose that must survive.\n\n## Log\n\n"
    return head + log


_FENCE = f"{COMPILED_TRUTH_START}\nAnna works at Acme.\n{COMPILED_TRUTH_END}"

_WORKS_AT = "relations:\n  - rel: works_at\n    target: organisations/acme\n"


def _dreamed_from(paths: VaultPaths, base: str) -> None:
    (paths.metadata / "dream.json").write_text(
        json.dumps({"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}),
        encoding="utf-8",
    )


def test_marker_state_absent_ok_and_broken() -> None:
    assert marker_state("no markers here") == "absent"
    assert marker_state(f"{COMPILED_TRUTH_START}\nx\n{COMPILED_TRUTH_END}") == "ok"
    assert marker_state(f"{COMPILED_TRUTH_START}\nx\n") == "broken"
    assert marker_state(f"{COMPILED_TRUTH_END}\nx\n") == "broken"
    assert marker_state(f"{COMPILED_TRUTH_END}\nx\n{COMPILED_TRUTH_START}") == "broken"
    assert marker_state(
        f"{COMPILED_TRUTH_START}\nx\n{COMPILED_TRUTH_END}\n{COMPILED_TRUTH_END}"
    ) == "broken"


def test_marker_state_ignores_markers_inside_a_code_fence() -> None:
    """N7: a note DOCUMENTING the fence (this skill's own step-6 example, an
    AGENTS.md excerpt) carries both markers in a ``` block. Counting those
    made the note read as fenced, and the tool would overwrite the example."""
    documented = (
        "---\ntitle: Anna\n---\n\n"
        "The block looks like this:\n\n"
        f"```markdown\n{COMPILED_TRUTH_START}\n...\n{COMPILED_TRUTH_END}\n```\n"
    )
    assert marker_state(documented) == "absent"
    assert marker_state(
        f"{COMPILED_TRUTH_START}\nreal\n{COMPILED_TRUTH_END}\n\n" + documented
    ) == "ok"


def test_compiled_truth_block_is_not_read_back_as_log_evidence(
    tmp_path: Path,
) -> None:
    """N2: an append-bootstrapped fence lands inside the note's last section,
    which on an entity note is ``## Log``. _log_bullets stops at the next
    HEADING, not at the marker, so the pass's own summary was counted as new
    Log evidence and paired with the bullets it summarises."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT,
        "- 2026-06-01 — Anna joined the Acme platform team ([[knowledge/x]])\n"
        "\n"
        f"{COMPILED_TRUTH_START}\n"
        "- Anna works at [[knowledge/organisations/acme]] since 2025-01-01.\n"
        "- 2026-06-01 — Anna joined the Acme platform team.\n"
        f"{COMPILED_TRUTH_END}\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"][0]["log_bullets"] == 1
    assert packet["compiled_truth"][0]["new_log_bullets"] == 1
    assert packet["contradictions"] == []


def test_packet_compiled_truth_candidate_for_note_without_a_block(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           _entity(_WORKS_AT, "- 2026-06-01 — joined the platform team ([[knowledge/x]])\n"))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"] == [{
        "node_id": "people/anna",
        "note_path": "knowledge/people/anna.md",
        "marker_state": "absent",
        "log_bullets": 1,
        "new_log_bullets": 1,
        "changed_notes": 0,
        "relations_changed": True,
        "open_relations": ["works_at -> organisations/acme"],
    }]
    assert packet["compiled_truth_blocked"] == []


def test_packet_compiled_truth_skips_a_current_block_with_no_new_log_lines(
    tmp_path: Path,
) -> None:
    """A quiet entity must not burn an edit: the block is already there and
    the ## Log gained nothing since the last dream."""
    paths = _vault(tmp_path)
    note = _entity(_WORKS_AT, "- 2026-06-01 — joined ([[knowledge/x]])\n", marker=_FENCE)
    _write(paths, "knowledge/people/anna.md", note)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", note + "\nA hand edit below the fence.\n")
    _commit(paths, "hand edit", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"] == []


def test_packet_compiled_truth_ranks_by_new_evidence_and_honours_its_budget(
    tmp_path: Path,
) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           _entity(log="- 2026-06-01 — one ([[knowledge/x]])\n"))
    _write(paths, "knowledge/people/bo.md", _entity(
        log="- 2026-06-01 — one ([[knowledge/x]])\n- 2026-06-02 — two ([[knowledge/y]])\n"))
    _commit(paths, "two people", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert [c["node_id"] for c in packet["compiled_truth"]] == ["people/bo", "people/anna"]
    capped = build_packet(
        paths, verdict=verdict, top_k=5, compiled_truth_budget=1, logger=_LOG,
    )
    assert [c["node_id"] for c in capped["compiled_truth"]] == ["people/bo"]
    assert capped["edit_budget"] == {
        "total_note_edits": 10,
        "compiled_truth": 1,
        "reserved_for_other_jobs": 9,
        "contradiction_proposals": 5,
    }


def test_packet_compiled_truth_blocks_an_ambiguous_fence(tmp_path: Path) -> None:
    """A START with no END has no safe write point — the note is reported,
    never compiled (concepts.py refuses the same shape)."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           _entity(log="- 2026-06-01 — one ([[knowledge/x]])\n", marker=COMPILED_TRUTH_START))
    _commit(paths, "half fence", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"] == []
    assert packet["compiled_truth_blocked"] == ["knowledge/people/anna.md"]


def test_packet_compiled_truth_resolves_a_folder_backed_project(tmp_path: Path) -> None:
    """A project's entity note is the overview INSIDE its folder, so the
    candidate's node id is the canonical projects/<slug>/<slug>."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/projects/brain/brain.md",
           _entity(log="- 2026-06-01 — shipped the gate ([[knowledge/x]])\n"))
    _write(paths, "knowledge/projects/brain/log/2026-06-11.md", "# log\n")
    _commit(paths, "project", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert [c["note_path"] for c in packet["compiled_truth"]] == [
        "knowledge/projects/brain/brain.md",
    ]
    assert packet["compiled_truth"][0]["node_id"] == "projects/brain/brain"


# --------------------------------------------------------------------------
# contradiction detection
# --------------------------------------------------------------------------

def _contradiction_packet(
    paths: VaultPaths, **kwargs: int
) -> list[ContradictionCandidate]:
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG, **kwargs)
    return packet["contradictions"]


def test_contradiction_relation_both_closed_and_continuously_open(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: works_at\n    target: organisations/acme\n    valid_from: '2025-01-01'\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-01-01'\n",
        "- 2026-06-01 — a neutral note ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_interval_conflict"]
    assert found[0]["node_id"] == "people/anna"
    assert "2026-01-01" in found[0]["detail"]


def test_contradiction_supersession_is_not_a_conflict(tmp_path: Path) -> None:
    """Close the old interval, open a new one after it — the documented
    supersede shape, and not a contradiction."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-01-01'\n"
        "  - rel: works_at\n    target: organisations/acme\n    valid_from: '2026-02-01'\n",
        "- 2026-06-01 — a neutral note ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_contradiction_open_relation_against_a_termination_cue_in_the_log(
    tmp_path: Path,
) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_vs_log"]
    assert found[0]["left"] == "works_at -> organisations/acme [(open start).. (open)]"
    assert found[0]["right"].startswith("- 2026-06-02 — Anna left Acme")
    assert "'left'" in found[0]["detail"]


def test_contradiction_ignores_a_log_line_about_another_target(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- 2026-06-02 — Anna left the steering group ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_contradiction_restated_fact_in_the_log(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(log=(
        "- 2026-06-01 — Anna joined the platform team at Acme ([[knowledge/x]])\n"
        "- 2026-06-03 — Anna joined the platform team at Acme Corp ([[knowledge/y]])\n"
    )))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["restated_fact"]
    assert found[0]["left"].startswith("- 2026-06-01")
    assert found[0]["right"].startswith("- 2026-06-03")


def test_contradiction_proposal_path_is_content_addressed_and_stable(
    tmp_path: Path,
) -> None:
    """Re-detecting the same contradiction must address the same fact note,
    and address it through the shared proposer rather than a private naming
    scheme, so every deterministic proposer agrees on one inbox filename."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    first = _contradiction_packet(paths)
    second = _contradiction_packet(paths)
    assert first == second
    finding = first[0]
    expected = propose_fact(
        paths,
        kind="dream-contradiction",
        title="anything",
        target=finding["node_id"],
        source="knowledge/people/anna",
        reason="anything",
        relations=[
            Relation(
                rel=e["rel"], target=e["target"],
                valid_from=e["valid_from"], valid_until=e["valid_until"],
                source=e["source"],
            )
            for e in finding["promote_relations"]
        ],
        fact=finding["proposal_fact"],
        dry_run=True,
    )
    assert expected.path == finding["proposal_path"]
    assert finding["proposal_path"].startswith("knowledge/assistant/inbox/")
    assert not (paths.root / finding["proposal_path"]).exists()


def test_contradiction_proposal_fact_carries_both_claims_as_one_log_line(
    tmp_path: Path,
) -> None:
    """promote.fact lands verbatim in the entity's Log on approval, so it
    states the disagreement (never a resolution) on a single unnested line."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    fact = _contradiction_packet(paths)[0]["proposal_fact"]
    assert "\n" not in fact
    assert '"- ' not in fact
    assert "relation_vs_log" in fact
    assert "works_at -> organisations/acme" in fact
    assert "Anna left Acme for a new role" in fact


def test_contradiction_already_in_the_inbox_frees_its_budget_slot(
    tmp_path: Path,
) -> None:
    """A standing finding must not occupy the same slot every night: once its
    proposal is in the human's queue the packet drops it and the next-most
    severe finding takes the place."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: member_of\n    target: organisations/kern\n    valid_from: '2025-01-01'\n"
        "  - rel: member_of\n    target: organisations/kern\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-01-01'\n",
        "- 2026-06-01 — Anna joined the platform team ([[knowledge/x]])\n"
        "- 2026-06-03 — Anna joined the platform squad ([[knowledge/y]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    top = _contradiction_packet(paths, contradiction_budget=1)[0]
    assert top["kind"] == "relation_interval_conflict"

    inbox = paths.root / "knowledge/assistant/inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (paths.root / top["proposal_path"]).write_text("---\ntitle: x\n---\n", encoding="utf-8")

    after = _contradiction_packet(paths, contradiction_budget=1)
    assert [c["kind"] for c in after] == ["restated_fact"]


def test_contradiction_already_promoted_out_of_the_inbox_is_not_reproposed(
    tmp_path: Path,
) -> None:
    """consolidate.py MOVES an approved fact note to
    knowledge/assistant/archive/<YYYY-MM>/; the inbox is empty again, and a
    detector that keeps firing must not re-propose what a human settled."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_vs_log"]
    archived = paths.root / "knowledge/assistant/archive/2026-06"
    archived.mkdir(parents=True, exist_ok=True)
    name = found[0]["proposal_path"].rsplit("/", 1)[-1]
    (archived / name).write_text("---\ntitle: x\n---\n", encoding="utf-8")
    assert _contradiction_packet(paths) == []


def test_contradictions_are_severity_ordered_and_capped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: member_of\n    target: organisations/kern\n    valid_from: '2025-01-01'\n"
        "  - rel: member_of\n    target: organisations/kern\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-01-01'\n",
        "- 2026-06-01 — Anna joined the platform team ([[knowledge/x]])\n"
        "- 2026-06-03 — Anna joined the platform squad ([[knowledge/y]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert [c["kind"] for c in _contradiction_packet(paths)] == [
        "relation_interval_conflict", "restated_fact",
    ]
    assert [c["kind"] for c in _contradiction_packet(paths, contradiction_budget=1)] == [
        "relation_interval_conflict",
    ]


def test_packet_with_entity_notes_stays_deterministic(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- 2026-06-02 — Anna left Acme ([[knowledge/x]])\n",
    ))
    _write(paths, "knowledge/people/bo.md", _entity(log="- 2026-06-02 — hi ([[knowledge/x]])\n"))
    _commit(paths, "entities", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    one = json.dumps(build_packet(paths, verdict=verdict, top_k=5, logger=_LOG), sort_keys=True)
    two = json.dumps(build_packet(paths, verdict=verdict, top_k=5, logger=_LOG), sort_keys=True)
    assert one == two


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


# --------------------------------------------------------------------------
# compiled truth: what counts as new evidence (M1, M2, m9)
# --------------------------------------------------------------------------

_ACME_UNTIL = (
    "relations:\n"
    "  - rel: works_at\n    target: organisations/acme\n"
    "    valid_from: '2025-01-01'\n    valid_until: '2026-06-02'\n"
)
_ACME_OPEN = (
    "relations:\n"
    "  - rel: works_at\n    target: organisations/acme\n"
    "    valid_from: '2025-01-01'\n"
)


def test_compiled_truth_refreshes_when_a_relation_closes(tmp_path: Path) -> None:
    """M1: closing a relation appends no ## Log bullet — upsert_relation_in_text
    edits frontmatter only — so a block gated on new bullets would assert the
    closed fact for ever."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _entity(
        _ACME_OPEN, "- 2026-06-01 — joined ([[knowledge/x]])\n", marker=_FENCE))
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _ACME_UNTIL, "- 2026-06-01 — joined ([[knowledge/x]])\n", marker=_FENCE))
    _commit(paths, "anna left", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert [c["node_id"] for c in packet["compiled_truth"]] == ["people/anna"]
    candidate = packet["compiled_truth"][0]
    assert candidate["new_log_bullets"] == 0
    assert candidate["relations_changed"] is True
    assert candidate["open_relations"] == []


def test_compiled_truth_open_relations_say_since_when(tmp_path: Path) -> None:
    """The block is asked to state since when; the packet renders it once."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _ACME_OPEN, "- 2026-06-01 — joined ([[knowledge/x]])\n"))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"][0]["open_relations"] == [
        "works_at -> organisations/acme (since 2025-01-01)",
    ]


def test_compiled_truth_counts_evidence_from_the_project_folder(tmp_path: Path) -> None:
    """M2: a real project's activity lands in knowledge/projects/<slug>/log/,
    not in the overview's own ## Log — the shape that made the live vault's
    projects permanently ineligible."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/brain/brain.md", _entity(
        log="- 2026-06-01 — shipped the gate ([[knowledge/x]])\n", marker=_FENCE))
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/projects/brain/log/2026-06-11.md",
           "---\ntitle: log\ntype: project\n---\n\n# 2026-06-11\n\n- did a thing\n")
    _commit(paths, "session log", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert [c["note_path"] for c in packet["compiled_truth"]] == [
        "knowledge/projects/brain/brain.md",
    ]
    candidate = packet["compiled_truth"][0]
    assert candidate["new_log_bullets"] == 0    # the overview's own Log is quiet
    assert candidate["changed_notes"] == 1


def test_compiled_truth_overview_without_a_log_qualifies_on_relation_change(
    tmp_path: Path,
) -> None:
    """M2: 11 of the live vault's 16 project overviews carry no ## Log at all;
    an overview with open relations must still qualify when they move."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/brain/brain.md",
           "---\ntitle: brain\ntype: project\n"
           "relations:\n  - rel: related_to\n    target: projects/kern/kern\n"
           "---\n\n# brain\n\nHand-written overview.\n")
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/projects/brain/brain.md",
           "---\ntitle: brain\ntype: project\n"
           "relations:\n  - rel: related_to\n    target: projects/kern/kern\n"
           "  - rel: related_to\n    target: projects/agent/agent\n"
           "---\n\n# brain\n\nHand-written overview.\n")
    _commit(paths, "new relation", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert [c["node_id"] for c in packet["compiled_truth"]] == ["projects/brain/brain"]
    assert packet["compiled_truth"][0]["log_bullets"] == 0
    assert packet["compiled_truth"][0]["relations_changed"] is True


def test_compiled_truth_skips_a_blockless_note_with_no_new_evidence(
    tmp_path: Path,
) -> None:
    """m9: 'something to compile AND new evidence' is the documented rule —
    marker_state: absent must not bypass the second half."""
    paths = _vault(tmp_path)
    note = _entity(_WORKS_AT, "- 2026-06-01 — joined ([[knowledge/x]])\n")
    _write(paths, "knowledge/people/anna.md", note)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           note.replace("Hand-written prose", "Reworded prose"))
    _commit(paths, "prose edit", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"] == []


# --------------------------------------------------------------------------
# the gate must not dream on the pass's own output
# --------------------------------------------------------------------------

def _proposal_note(target: str) -> str:
    return (
        "---\n"
        f"title: 'Candidate relation on {target}'\n"
        "type: memory_fact\n"
        "author: 'script:wikilink-relation'\n"
        "written_via: script\n"
        "memory_status: unconsolidated\n"
        "confirmations: 0\n"
        "approved: false\n"
        "promote:\n"
        f"  target: {target}\n"
        "  relations: []\n"
        "  fact: ''\n"
        "  source: knowledge/people/anna\n"
        "---\n\nBecause a wikilink says so.\n"
    )


def test_gate_ignores_script_written_inbox_proposals(tmp_path: Path) -> None:
    """refute-propose M2: 308 machine-written candidates counted as 308
    'changes' fired a dream run on the proposer's own noise. A proposal is a
    question for a human, not new information."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    _dreamed_from(paths, base)
    for i in range(6):    # over the default threshold
        _write(paths, f"knowledge/assistant/inbox/wikilink-relation--p{i}--{i}.md",
               _proposal_note(f"people/p{i}"))
    _commit(paths, "proposals", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert verdict.changed_notes == ()
    assert not verdict.should_dream


def test_gate_still_counts_a_human_written_inbox_fact(tmp_path: Path) -> None:
    """Only the script-written half is excluded: a fact an agent or a human
    put in the inbox is real new information."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/assistant/inbox/hand-written.md",
           _proposal_note("people/anna").replace("written_via: script\n", ""))
    _commit(paths, "inbox fact", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    assert verdict.changed_notes == ("knowledge/assistant/inbox/hand-written.md",)


def test_gate_ignores_a_compiled_truth_only_edit(tmp_path: Path) -> None:
    """m10: the compiled-truth job rewrites hand-written entity notes, which
    carry no generated_by — up to three self-authored 'changes' a night
    counting toward a threshold of five."""
    paths = _vault(tmp_path)
    note = _entity(_WORKS_AT, "- 2026-06-01 — joined ([[knowledge/x]])\n", marker=_FENCE)
    _write(paths, "knowledge/people/anna.md", note)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           note.replace("Anna works at Acme.", "Anna has worked at Acme since 2025."))
    _commit(paths, "dream refresh", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    assert verdict.changed_notes == ()


def test_gate_counts_a_note_edited_outside_its_compiled_truth(tmp_path: Path) -> None:
    """The exclusion is exact: a Log line added in the same commit as a block
    refresh is still new information."""
    paths = _vault(tmp_path)
    note = _entity(_WORKS_AT, "- 2026-06-01 — joined ([[knowledge/x]])\n", marker=_FENCE)
    _write(paths, "knowledge/people/anna.md", note)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           note.replace("Anna works at Acme.", "Anna has worked at Acme since 2025.")
               + "- 2026-06-11 — spoke at the summit ([[knowledge/y]])\n")
    _commit(paths, "refresh plus a fact", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    assert verdict.changed_notes == ("knowledge/people/anna.md",)


# --------------------------------------------------------------------------
# contradiction detection: false positives and self-perpetuation
# --------------------------------------------------------------------------

def test_contradiction_ignores_point_in_time_relations(tmp_path: Path) -> None:
    """M3: `attended` and `met_at` are point-in-time in the closed vocabulary,
    so such an edge is always open and correctly so. Meetings end; one live
    person note carries 28 of these."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n  - rel: attended\n    target: meetings/2026/2026-06-16-intro\n"
        "  - rel: met_at\n    target: meetings/2026/2026-06-16-intro\n",
        "- 2026-06-16 — the 2026-06-16-intro meeting ended early ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_contradiction_proposal_carries_the_closure_it_implies(tmp_path: Path) -> None:
    """M4: approving the proposal must SETTLE the contradiction. The closure
    rides in promote.relations, so consolidate closes the open span through
    upsert_relation_in_text — supersede, never delete."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _ACME_OPEN, "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_vs_log"]
    assert found[0]["promote_relations"] == [{
        "rel": "works_at",
        "target": "organisations/acme",
        "valid_from": "2025-01-01",
        "valid_until": "2026-06-02",
        "source": "",
    }]


def test_contradiction_without_a_dated_log_line_proposes_no_closure(
    tmp_path: Path,
) -> None:
    """No date on the bullet, nothing to close the relation ON: the proposal
    carries the disagreement alone rather than inventing an end date."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _WORKS_AT, "- Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_vs_log"]
    assert found[0]["promote_relations"] == []


def test_closure_earlier_than_the_relations_start_is_not_a_closure(
    tmp_path: Path,
) -> None:
    """N4: the rejoin shape — left in June, came back in August, the old span
    already closed. The stale Log line still trips the detector, but writing
    its date as the NEW span's valid_until hands consolidate an incoherent
    ``from 2026-08-01 until 2026-06-02``."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-06-02'\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        "    valid_from: '2026-08-01'\n",
        "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")

    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_vs_log"]
    assert found[0]["promote_relations"] == []
    assert (
        "no closure proposed: the Log line's date 2026-06-02 precedes the "
        "relation's own start 2026-08-01"
    ) in found[0]["proposal_fact"]


def test_contradiction_ignores_its_own_promoted_flag_line(tmp_path: Path) -> None:
    """M4: consolidate appends promote.fact to the ## Log, and that line
    quotes the offending bullet — target mention and termination cue intact.
    Without this the detector re-fires on its own promoted text, against
    different wording, so neither the inbox nor the archive check catches it."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    promoted = (
        f"- 2026-06-12 — {CONTRADICTION_FLAG_PREFIX} (relation_vs_log): "
        '"works_at -> organisations/acme [(open start).. (open)]" vs '
        '"2026-06-02 — Anna left Acme for a new role" ([[knowledge/y]])\n'
    )
    _write(paths, "knowledge/people/anna.md", _entity(_WORKS_AT, promoted))
    _commit(paths, "promoted flag", when="2026-06-11T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_contradiction_target_slug_matches_on_word_boundaries(tmp_path: Path) -> None:
    """m7/n14: live project slugs include `check`, `agent`, `brain` and
    `server`; an unanchored substring test fires on 'checklist'."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n  - rel: collaborator_on\n    target: projects/check/check\n",
        "- 2026-06-02 — wrote a checklist and no longer chases the old draft "
        "([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_contradiction_target_slug_still_matches_as_a_word(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n  - rel: collaborator_on\n    target: projects/check/check\n",
        "- 2026-06-02 — Anna left check for another team ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert [c["kind"] for c in _contradiction_packet(paths)] == ["relation_vs_log"]


def test_two_findings_addressing_one_file_take_one_budget_slot(
    tmp_path: Path,
) -> None:
    """m8: a duplicated relation entry detects the same conflict twice; both
    hits address the same content-hashed file, so the packet must not spend
    two slots (and list one path twice)."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: works_at\n    target: organisations/acme\n    valid_from: '2025-01-01'\n"
        "  - rel: works_at\n    target: organisations/acme\n    valid_from: '2025-01-01'\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-01-01'\n",
        "- 2026-06-01 — a neutral note ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    found = _contradiction_packet(paths)
    assert [c["kind"] for c in found] == ["relation_interval_conflict"]
    assert len({c["proposal_path"] for c in found}) == 1


def test_interval_conflict_survives_a_malformed_valid_from(tmp_path: Path) -> None:
    """n18: a non-date valid_from compared lexically against the closure date
    could silently suppress (or invent) the conflict."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        "relations:\n"
        "  - rel: works_at\n    target: organisations/acme\n    valid_from: last spring\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        "    valid_from: '2025-01-01'\n    valid_until: '2026-01-01'\n",
        "- 2026-06-01 — a neutral note ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert [c["kind"] for c in _contradiction_packet(paths)] == [
        "relation_interval_conflict",
    ]


def test_restated_facts_ignore_function_words(tmp_path: Path) -> None:
    """m12: with 'the'/'and'/'for' in the token set, two short bullets about
    different things clear the 0.6 bar."""
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(log=(
        "- 2026-06-01 — ran the dream pass ([[knowledge/x]])\n"
        "- 2026-06-03 — ran the dream gate ([[knowledge/y]])\n"
    )))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_restated_facts_are_capped_per_note() -> None:
    """m11: the window bounds the INPUTS; without an output cap a Log of
    near-identical bullets emits O(n^2) findings before the budget cut."""
    bullets = [
        f"- 2026-06-01 — Anna joined the platform team at Acme variant{i:03d} "
        "([[knowledge/x]])"
        for i in range(20)
    ]
    found = _restated_facts("people/anna", "knowledge/people/anna.md", bullets)
    assert len(found) == 50


# --------------------------------------------------------------------------
# M5: the deterministic half writes the survivors
# --------------------------------------------------------------------------

def _anna_left_acme(paths: VaultPaths) -> ContradictionCandidate:
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md", _entity(
        _ACME_OPEN, "- 2026-06-02 — Anna left Acme for a new role ([[knowledge/x]])\n",
    ))
    _commit(paths, "anna", when="2026-06-11T12:00:00+00:00")
    return _contradiction_packet(paths)[0]


def _survivor(finding: ContradictionCandidate) -> AdjudicatedFinding:
    return AdjudicatedFinding(
        node_id=finding["node_id"],
        note_path=finding["note_path"],
        proposal_path=finding["proposal_path"],
        proposal_fact=finding["proposal_fact"],
        promote_relations=finding["promote_relations"],
        title="Contradiction: Anna's Acme employment",
        body="The note says she left; the relation is still open.",
    )


def test_write_proposals_stamps_the_memory_fact_contract(tmp_path: Path) -> None:
    """M5: AGENTS.md says propose.py is the dream pass's contradiction
    writer. It is now — so every proposal carries approved: false and the
    script provenance only a script may claim."""
    paths = _vault(tmp_path)
    finding = _anna_left_acme(paths)

    outcomes = write_proposals(
        paths, [_survivor(finding)], now=AS_OF, logger=_LOG,
    )

    assert outcomes == [{"proposal_path": finding["proposal_path"], "action": "proposed"}]
    written = (paths.root / finding["proposal_path"]).read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(written)
    assert frontmatter["written_via"] == "script"
    assert frontmatter["author"] == "script:dream-contradiction"
    assert frontmatter["approved"] is False
    assert frontmatter["memory_status"] == "unconsolidated"
    assert frontmatter["confirmations"] == 0
    promote = frontmatter["promote"]
    assert isinstance(promote, dict)
    assert promote["target"] == "people/anna"
    assert promote["fact"] == finding["proposal_fact"]
    assert promote["relations"] == [{
        "rel": "works_at", "target": "organisations/acme",
        "valid_from": "2025-01-01", "valid_until": "2026-06-02",
    }]
    assert "The note says she left" in body


def test_write_proposals_is_idempotent(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    survivor = _survivor(_anna_left_acme(paths))
    write_proposals(paths, [survivor], now=AS_OF, logger=_LOG)
    again = write_proposals(paths, [survivor], now=AS_OF, logger=_LOG)
    assert [o["action"] for o in again] == ["exists"]


def test_write_proposals_refuses_a_reworded_fact(tmp_path: Path) -> None:
    """The proposal's path is a hash of its fact and relations; a reworded
    fact would land somewhere else and defeat every dedup the packet does."""
    paths = _vault(tmp_path)
    survivor = _survivor(_anna_left_acme(paths))
    survivor["proposal_fact"] = survivor["proposal_fact"] + " (tidied up)"
    with pytest.raises(ValueError, match="addresses"):
        write_proposals(paths, [survivor], now=AS_OF, logger=_LOG)
    assert list((paths.root / "knowledge/assistant/inbox").glob("*.md")) == []


def test_approving_a_proposal_closes_the_relation(tmp_path: Path) -> None:
    """M4 end to end: approval must SETTLE the contradiction. consolidate
    merges promote.relations, so the open span gets its valid_until and the
    fact line lands in the ## Log — the whole point of proposing."""
    paths = _vault(tmp_path)
    survivor = _survivor(_anna_left_acme(paths))
    write_proposals(paths, [survivor], now=AS_OF, logger=_LOG)
    proposal = paths.root / survivor["proposal_path"]
    proposal.write_text(
        proposal.read_text(encoding="utf-8").replace("approved: false", "approved: true"),
        encoding="utf-8",
    )

    consolidate(paths, logger=_LOG, as_of=date(2026, 6, 12))

    note = (paths.root / "knowledge/people/anna.md").read_text(encoding="utf-8")
    relations, problems = parse_relations(_split_frontmatter(note)[0])
    assert problems == []
    assert [(r.rel, r.target, r.valid_until) for r in relations] == [
        ("works_at", "organisations/acme", "2026-06-02"),
    ]
    assert CONTRADICTION_FLAG_PREFIX in note

    # ... and the settled contradiction does not come back: the relation is
    # closed, and the flag line it left behind is not read as evidence.
    _commit(paths, "consolidated", when="2026-06-12T12:00:00+00:00")
    assert _contradiction_packet(paths) == []


def test_parse_adjudicated_refuses_a_malformed_payload() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        parse_adjudicated({"node_id": "people/anna"})
    with pytest.raises(ValueError, match="'title'"):
        parse_adjudicated([{
            "node_id": "people/anna",
            "note_path": "knowledge/people/anna.md",
            "proposal_path": "knowledge/assistant/inbox/x.md",
            "proposal_fact": "they disagree",
            "promote_relations": [],
            "body": "because",
        }])
    with pytest.raises(ValueError, match="'rel'"):
        parse_adjudicated([{
            "node_id": "people/anna",
            "note_path": "knowledge/people/anna.md",
            "proposal_path": "knowledge/assistant/inbox/x.md",
            "proposal_fact": "they disagree",
            "promote_relations": [{"target": "organisations/acme"}],
            "title": "t",
            "body": "because",
        }])


def test_parse_adjudicated_accepts_the_packet_shape(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    finding = _anna_left_acme(paths)
    payload = json.loads(json.dumps([_survivor(finding)]))
    assert parse_adjudicated(payload) == [_survivor(finding)]


def test_compiled_truth_first_block_for_a_project_with_only_folder_evidence(
    tmp_path: Path,
) -> None:
    """M2, the live shape: knowledge/projects/brain/brain.md carries no
    ## Log and no relations — all of its activity is in log/<date>.md. It
    must still get a first block, or the job is inert where it matters."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/brain/brain.md",
           "---\ntitle: brain\ntype: project\n---\n\n# brain\n\n## Overview\n\nA vault.\n")
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/projects/brain/log/2026-06-11.md",
           "---\ntitle: log\ntype: project\n---\n\n# 2026-06-11\n\n- shipped the gate\n")
    _commit(paths, "session log", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert [c["node_id"] for c in packet["compiled_truth"]] == ["projects/brain/brain"]
    assert packet["compiled_truth"][0]["marker_state"] == "absent"


def test_compiled_truth_skips_an_entity_with_nothing_to_compile(
    tmp_path: Path,
) -> None:
    """A person note that changed but has no Log, no relations and no notes
    under it: there is nothing a first block could say."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md",
           "---\ntitle: Anna\ntype: person\n---\n\n# Anna\n\nA person.\n")
    base = _commit(paths, "base", when="2026-06-09T12:00:00+00:00")
    _dreamed_from(paths, base)
    _write(paths, "knowledge/people/anna.md",
           "---\ntitle: Anna\ntype: person\n---\n\n# Anna\n\nA person I met.\n")
    _commit(paths, "prose", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    packet = build_packet(paths, verdict=verdict, top_k=5, logger=_LOG)
    assert packet["compiled_truth"] == []
