"""Dream-pass contradiction detection (split out of test_dream.py,
AUD-121): relation-interval conflicts, relation-vs-log and restated-fact
findings (plus their false-positive guards), and M5's deterministic
proposal writer that survives an adjudication round trip.

Self-contained (its own copy of the small vault/entity helpers) rather
than importing test_dream.py or a shared sibling, since mypy's mypy_path
does not cover tests/ and can't resolve a bare sibling import there."""
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
    CONTRADICTION_FLAG_PREFIX,
    ContradictionCandidate,
    build_packet,
    evaluate_gate,
    parse_adjudicated,
    write_proposals,
)
from ingest_lib.dream import _restated_facts  # private: the emission cap is its own
from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import parse_relations
from ingest_lib.propose import propose_fact
from ingest_lib.relations import Relation

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


_WORKS_AT = "relations:\n  - rel: works_at\n    target: organisations/acme\n"

_ACME_OPEN = (
    "relations:\n"
    "  - rel: works_at\n    target: organisations/acme\n"
    "    valid_from: '2025-01-01'\n"
)


def _dreamed_from(paths: VaultPaths, base: str) -> None:
    (paths.metadata / "dream.json").write_text(
        json.dumps({"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}),
        encoding="utf-8",
    )


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
