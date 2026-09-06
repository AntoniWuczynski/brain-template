"""Tests for ingest_lib.duplicates (merge proposals for sweep's
entity-duplicate pairs). No LLM, no wall-clock quirks — mirrors
test_propose.py's wikilinks section for shape."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import TypedDict, cast

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.consolidate import consolidate
from ingest_lib.duplicates import _with_merge_block, propose_duplicate_merges
from ingest_lib.notes import _split_frontmatter
from ingest_lib.sweep import find_duplicate_entity_pairs

_INBOX = "knowledge/assistant/inbox"


class _PromoteBlock(TypedDict):
    """The ``promote:`` mapping ``consolidate.py`` reads back off a proposal.

    ``merge`` is the duplicate-entity shape this pass exists to propose —
    every note it writes carries one (FOUNDER_DECISIONS.md IMP-021)."""

    target: str
    relations: list[dict[str, str]]
    fact: str
    source: str
    merge: dict[str, str]


def _promote(frontmatter: Mapping[str, object]) -> _PromoteBlock:
    """The ``promote:`` block, shape-checked once so every assertion below
    reads a typed field instead of indexing YAML's untyped mapping."""
    block = frontmatter["promote"]
    assert isinstance(block, dict)
    assert isinstance(block.get("target"), str)
    assert isinstance(block.get("fact"), str)
    assert isinstance(block.get("source"), str)
    assert isinstance(block.get("relations"), list)
    assert isinstance(block.get("merge"), dict)
    return cast(_PromoteBlock, block)


EMAIL_PERSON = """---
title: dana@example.com
type: person
aliases: []
---

# dana@example.com
"""

NAMED_PERSON = """---
title: Dana Scott
type: person
aliases: []
---

# Dana Scott
"""


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> Path:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _inbox_notes(paths: VaultPaths) -> list[Path]:
    d = paths.root / _INBOX
    return sorted(d.glob("*.md")) if d.is_dir() else []


def _seed_pair(paths: VaultPaths) -> None:
    _write(paths, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    _write(paths, "knowledge/people/dana-scott.md", NAMED_PERSON)


def test_propose_duplicate_merges_writes_a_candidate(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)

    results = propose_duplicate_merges(paths)
    assert [r.action for r in results] == ["proposed"]

    notes = _inbox_notes(paths)
    assert len(notes) == 1
    fm, body = _split_frontmatter(notes[0].read_text(encoding="utf-8"))
    promote = _promote(fm)
    assert promote["target"] == "people/dana-scott"
    # A vault-relative path, not a bare node id: it lands inside the [[…]]
    # of the survivor's Log line, where a node id would dangle.
    assert promote["source"] == "knowledge/people/danaexamplecom"
    assert promote["relations"] == []
    assert promote["merge"] == {
        "duplicate": "people/danaexamplecom", "survivor": "people/dana-scott",
    }
    assert "people/dana-scott" in promote["fact"]
    assert "superseded_by" in promote["fact"]
    # The body still spells out what approval will do.
    assert "What approving this note will do" in body
    assert "close every open relation on the duplicate with valid_until" in body


def test_propose_duplicate_merges_rerun_is_a_noop(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)

    propose_duplicate_merges(paths)
    before = {p.name: p.read_bytes() for p in _inbox_notes(paths)}

    second = propose_duplicate_merges(paths)
    assert [r.action for r in second] == ["exists"]
    after = {p.name: p.read_bytes() for p in _inbox_notes(paths)}
    assert after == before


def test_propose_duplicate_merges_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)

    results = propose_duplicate_merges(paths, dry_run=True)
    assert [r.action for r in results] == ["proposed"]
    assert not (paths.root / _INBOX).exists()


def test_propose_duplicate_merges_stops_once_pair_is_merged(tmp_path: Path) -> None:
    """Same suppression sweep uses: once the address is an alias on the
    named node, no proposal is written."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    _write(
        paths, "knowledge/people/dana-scott.md",
        "---\ntitle: Dana Scott\ntype: person\naliases: [dana@example.com]\n"
        "---\n\n# Dana Scott\n",
    )

    results = propose_duplicate_merges(paths)
    assert results == []
    assert not (paths.root / _INBOX).exists()


def test_propose_duplicate_merges_no_pairs_is_empty(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/dana-scott.md", NAMED_PERSON)

    assert propose_duplicate_merges(paths) == []


def test_propose_duplicate_merges_skips_ambiguous_pairs(tmp_path: Path) -> None:
    """Two named nodes ('Andrew' and 'Andrew Basley') both name the local
    part of one email-slugged node: there is no single survivor to name as
    promote.target, so this pass proposes nothing and leaves it for a human
    (the sweep report still flags the pair)."""
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/andrewexamplecom.md",
        "---\ntitle: andrew@example.com\ntype: person\naliases: []\n---\n\n"
        "# andrew@example.com\n",
    )
    _write(
        paths, "knowledge/people/andrew.md",
        "---\ntitle: Andrew\ntype: person\naliases: []\n---\n\n# Andrew\n",
    )
    _write(
        paths, "knowledge/people/andrew-basley.md",
        "---\ntitle: Andrew Basley\ntype: person\naliases: []\n---\n\n"
        "# Andrew Basley\n",
    )

    assert propose_duplicate_merges(paths) == []
    assert not (paths.root / _INBOX).exists()


def test_merge_block_is_self_healed_on_rerun(tmp_path: Path) -> None:
    """``propose_fact`` renders the four shared promote keys; this pass
    splices ``promote.merge`` in afterwards. A note left without it — a
    crash between the two writes — gains it on the next run instead of
    sitting in the inbox as an unexecutable proposal forever."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    propose_duplicate_merges(paths)
    note = _inbox_notes(paths)[0]
    stripped = "\n".join(
        line for line in note.read_text(encoding="utf-8").split("\n")
        if not line.startswith(("  merge:", "    duplicate:", "    survivor:"))
    )
    note.write_text(stripped, encoding="utf-8")
    promote_before = _split_frontmatter(stripped)[0]["promote"]
    assert isinstance(promote_before, dict) and "merge" not in promote_before

    second = propose_duplicate_merges(paths)

    assert [r.action for r in second] == ["reconciled"]
    fm, _body = _split_frontmatter(note.read_text(encoding="utf-8"))
    assert _promote(fm)["merge"] == {
        "duplicate": "people/danaexamplecom", "survivor": "people/dana-scott",
    }


def test_approved_proposal_merges_and_sweep_goes_quiet(tmp_path: Path) -> None:
    """End to end: propose -> a human approves -> consolidate performs the
    AGENTS.md merge -> sweep's own entity-duplicate detection stops
    reporting the pair, because the merged shape is now present."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    assert len(find_duplicate_entity_pairs(paths)) == 1

    propose_duplicate_merges(paths)
    note = _inbox_notes(paths)[0]
    note.write_text(
        note.read_text(encoding="utf-8").replace("approved: false", "approved: true", 1),
        encoding="utf-8",
    )

    stats = consolidate(paths, logger=logging.getLogger("test"), as_of=date(2026, 9, 6))

    assert stats.promoted == 1
    assert stats.problems == ()
    dup_fm, _ = _split_frontmatter(
        (paths.root / "knowledge/people/danaexamplecom.md").read_text(encoding="utf-8")
    )
    assert dup_fm["superseded_by"] == "knowledge/people/dana-scott"
    named_fm, _ = _split_frontmatter(
        (paths.root / "knowledge/people/dana-scott.md").read_text(encoding="utf-8")
    )
    assert named_fm["aliases"] == ["dana@example.com"]

    assert find_duplicate_entity_pairs(paths) == []
    # And the pass proposes nothing further for the merged pair.
    assert propose_duplicate_merges(paths) == []


# ---------------------------------------------------------------------------
# reconciliation: one pair is one note in the human's queue
#
# The live inbox is the reason this exists. The first version of this pass
# wrote proposals with no ``promote.merge`` and a bare NODE ID as
# ``promote.source``; re-wording ``_fact``/``_reason`` re-hashed the content
# key, so the next nightly run did not recognise those notes and minted a
# second one per pair. LIVE_SHAPED_PROPOSAL below is that shape, byte for
# byte (the live notes anonymised), and it must be healed in place.
# ---------------------------------------------------------------------------

# The body is one long line in the live notes; kept that way (assembled, so
# the source line stays readable) because reconciling must not reflow it.
_LIVE_SHAPED_BODY = (
    "Deterministic duplicate-entity pass "
    "(scripts/ingest_lib/duplicates.py), reusing sweep's `entity-duplicate` "
    "detection: [[knowledge/people/danaexamplecom]] is titled a bare email "
    "address and [[knowledge/people/dana-scott]] names the same local part.\n"
)

LIVE_SHAPED_PROPOSAL = f"""---
title: 'duplicate-entity merge candidate: people/danaexamplecom -> people/dana-scott'
type: memory_fact
created: '2026-09-06T01:00:06Z'
author: script:entity-duplicate-merge
written_via: script
memory_status: unconsolidated
confirmations: 2
approved: false
promote:
  target: people/dana-scott
  relations: []
  fact: '''dana@example.com'' (people/danaexamplecom) appears to be the same
    entity as people/dana-scott — merge them by superseding, never deleting.'
  source: people/danaexamplecom
---

{_LIVE_SHAPED_BODY}"""

# The name the FIRST version of the pass hashed this proposal to. The digest
# is deliberately not one today's code would compute: a reconciliation that
# only recognised the current filename would be no reconciliation at all.
LIVE_SHAPED_NAME = "entity-duplicate-merge--people-dana-scott--37bd4f1398fce39b.md"


def test_existing_proposal_is_reconciled_in_place_not_duplicated(
    tmp_path: Path,
) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(paths, f"{_INBOX}/{LIVE_SHAPED_NAME}", LIVE_SHAPED_PROPOSAL)

    results = propose_duplicate_merges(paths)

    assert [r.action for r in results] == ["reconciled"]
    notes = _inbox_notes(paths)
    assert [n.name for n in notes] == [LIVE_SHAPED_NAME]
    fm, body = _split_frontmatter(notes[0].read_text(encoding="utf-8"))
    promote = _promote(fm)
    assert promote["merge"] == {
        "duplicate": "people/danaexamplecom", "survivor": "people/dana-scott",
    }
    # The bare node id would dangle inside the [[…]] of the survivor's Log
    # line; the canonical vault-relative path does not.
    assert promote["source"] == "knowledge/people/danaexamplecom"
    # Everything the human owns is left exactly as it was.
    assert fm["approved"] is False
    assert fm["confirmations"] == 2
    assert fm["created"] == "2026-09-06T01:00:06Z"
    before_fm, before_body = _split_frontmatter(LIVE_SHAPED_PROPOSAL)
    before_promote = before_fm["promote"]
    assert isinstance(before_promote, dict)
    assert promote["fact"] == before_promote["fact"]
    assert body == before_body


def test_reconciled_proposal_is_left_alone_on_the_next_run(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(paths, f"{_INBOX}/{LIVE_SHAPED_NAME}", LIVE_SHAPED_PROPOSAL)
    propose_duplicate_merges(paths)
    before = {p.name: p.read_bytes() for p in _inbox_notes(paths)}

    second = propose_duplicate_merges(paths)

    assert [r.action for r in second] == ["exists"]
    assert {p.name: p.read_bytes() for p in _inbox_notes(paths)} == before


def test_two_proposals_for_one_pair_are_reconciled_without_minting_a_third(
    tmp_path: Path,
) -> None:
    """The state last night's run would have left behind. Both notes are
    healed and neither is deleted (a proposal is the human's to dispose of),
    but no third one is written."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(paths, f"{_INBOX}/{LIVE_SHAPED_NAME}", LIVE_SHAPED_PROPOSAL)
    _write(
        paths,
        f"{_INBOX}/entity-duplicate-merge--people-dana-scott--5860970ac31e2d97.md",
        LIVE_SHAPED_PROPOSAL,
    )

    results = propose_duplicate_merges(paths)

    assert [r.action for r in results] == ["reconciled", "reconciled"]
    notes = _inbox_notes(paths)
    assert len(notes) == 2
    for note in notes:
        promote = _promote(_split_frontmatter(note.read_text(encoding="utf-8"))[0])
        assert promote["merge"]["duplicate"] == "people/danaexamplecom"
        assert promote["source"] == "knowledge/people/danaexamplecom"


def test_reconcile_ignores_a_proposal_about_a_different_duplicate(
    tmp_path: Path,
) -> None:
    """Two email-slugged nodes can name the same person. The filename only
    carries the survivor, so a proposal about the OTHER duplicate must not be
    mistaken for this pair's — it is left alone and this pair still gets its
    own note."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    other = LIVE_SHAPED_PROPOSAL.replace("danaexamplecom", "danaoldmailcom")
    _write(
        paths,
        f"{_INBOX}/entity-duplicate-merge--people-dana-scott--0123456789abcdef.md",
        other,
    )

    results = propose_duplicate_merges(paths)

    assert [r.action for r in results] == ["proposed"]
    assert len(_inbox_notes(paths)) == 2
    untouched = (
        paths.root / _INBOX
        / "entity-duplicate-merge--people-dana-scott--0123456789abcdef.md"
    ).read_text(encoding="utf-8")
    assert untouched == other


def test_reconcile_dry_run_reports_but_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(paths, f"{_INBOX}/{LIVE_SHAPED_NAME}", LIVE_SHAPED_PROPOSAL)

    results = propose_duplicate_merges(paths, dry_run=True)

    assert [r.action for r in results] == ["reconciled"]
    assert [n.name for n in _inbox_notes(paths)] == [LIVE_SHAPED_NAME]
    assert (
        (paths.root / _INBOX / LIVE_SHAPED_NAME).read_text(encoding="utf-8")
        == LIVE_SHAPED_PROPOSAL
    )


def test_reconciled_proposal_promotes_the_merge_and_links_canonically(
    tmp_path: Path,
) -> None:
    """End to end over the live shape: reconcile, a human approves, and
    consolidate performs the AGENTS.md merge — with a Log link that resolves
    (the pre-reconciliation ``source:`` would have written a dangling one)."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(paths, f"{_INBOX}/{LIVE_SHAPED_NAME}", LIVE_SHAPED_PROPOSAL)
    propose_duplicate_merges(paths)
    note = paths.root / _INBOX / LIVE_SHAPED_NAME
    note.write_text(
        note.read_text(encoding="utf-8").replace("approved: false", "approved: true", 1),
        encoding="utf-8",
    )

    stats = consolidate(paths, logger=logging.getLogger("test"), as_of=date(2026, 9, 6))

    assert (stats.promoted, stats.problems) == (1, ())
    survivor = (paths.root / "knowledge/people/dana-scott.md").read_text(encoding="utf-8")
    assert "([[knowledge/people/danaexamplecom]])" in survivor
    dup_fm, _ = _split_frontmatter(
        (paths.root / "knowledge/people/danaexamplecom.md").read_text(encoding="utf-8")
    )
    assert dup_fm["superseded_by"] == "knowledge/people/dana-scott"
    assert find_duplicate_entity_pairs(paths) == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("no frontmatter at all\n", "no fence"),
        ("---\ntitle: x\n---\n\nbody\n", "no promote: key"),
        (
            "---\npromote:\n  merge:\n    duplicate: people/danaexamplecom\n"
            "    survivor: people/dana-scott\n---\n\nbody\n",
            "merge already present",
        ),
    ],
    ids=["no-fence", "no-promote-key", "merge-present"],
)
def test_merge_block_splice_leaves_these_notes_untouched(
    text: str, reason: str
) -> None:
    """``_with_merge_block``'s three early returns: it only ever edits a note
    of the shape this pass writes, and never twice."""
    assert _with_merge_block(
        text, duplicate="people/danaexamplecom", survivor="people/dana-scott"
    ) == text, reason
