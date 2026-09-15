"""Dream-pass gate: state IO, pending marker, gate arithmetic, packet
determinism. All vaults are tmp_path fixtures with a real git repo so the
gate's git arithmetic runs against genuine history; commit dates are
pinned via GIT_AUTHOR_DATE/GIT_COMMITTER_DATE for determinism.

Split (AUD-121) into sibling modules along its existing section
boundaries: test_dream_skills.py carries the SKILL.md/dream.sh contract
checks, test_dream_compiled.py the compiled-truth tests, and
test_dream_contradictions.py the contradiction-detection and M5
proposal-writer tests — each self-contained (its own copy of the small
vault/entity helpers) rather than importing this module or a shared one,
since mypy's mypy_path does not cover tests/ and can't resolve a bare
sibling import there. This module keeps the gate arithmetic, packet
basics, and the gate's own-output exclusions."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.dream import (
    COMPILED_TRUTH_END,
    COMPILED_TRUTH_START,
    DreamState,
    build_packet,
    evaluate_gate,
    load_pending_since,
    load_state,
    mark_done,
    record_pending,
)
from ingest_lib.hashing import sha256_of
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
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
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
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
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
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
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
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _write(paths, "knowledge/people/anna.md",
           note.replace("Anna works at Acme.", "Anna has worked at Acme since 2025.")
               + "- 2026-06-11 — spoke at the summit ([[knowledge/y]])\n")
    _commit(paths, "refresh plus a fact", when="2026-06-11T12:00:00+00:00")

    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=1)
    assert verdict.changed_notes == ("knowledge/people/anna.md",)
