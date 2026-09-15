"""Dream-pass compiled truth (split out of test_dream.py, AUD-121):
marker_state parsing, what counts as new evidence (M1, M2, m9), and the
packet's compiled_truth candidate ranking/budget.

Self-contained (its own copy of the small vault/entity helpers) rather
than importing test_dream.py or a shared sibling, since mypy's mypy_path
does not cover tests/ and can't resolve a bare sibling import there."""
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
    build_packet,
    evaluate_gate,
    marker_state,
)

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

_ACME_OPEN = (
    "relations:\n"
    "  - rel: works_at\n    target: organisations/acme\n"
    "    valid_from: '2025-01-01'\n"
)

_ACME_UNTIL = (
    "relations:\n"
    "  - rel: works_at\n    target: organisations/acme\n"
    "    valid_from: '2025-01-01'\n    valid_until: '2026-06-02'\n"
)


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
