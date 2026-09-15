"""Tests for ingest_lib.propose (the reusable fact-note proposer) and
ingest_lib.wikilinks (the deterministic wikilink-derived relation pass
built on top of it). No LLM, no wall-clock: every propose call injects
``now`` and every assertion is on file contents and stats."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict, cast

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.consolidate import consolidate
from ingest_lib.notes import _split_frontmatter
from ingest_lib.propose import ProposeResult, propose_fact
from ingest_lib.relations import Relation, parse_relations
from ingest_lib.wikilinks import find_candidates, propose_wikilink_relations

NOW = datetime(2026, 6, 12, 9, 30, 0, tzinfo=UTC)
_INBOX = "knowledge/assistant/inbox"
_ARCHIVE = "knowledge/assistant/archive"
_LOG = logging.getLogger("test_propose")


class _PromoteBlock(TypedDict):
    """The ``promote:`` mapping ``consolidate.py`` reads back off a proposal."""

    target: str
    relations: list[dict[str, str]]
    fact: str
    source: str


def _promote(frontmatter: Mapping[str, object]) -> _PromoteBlock:
    """The ``promote:`` block, shape-checked once so every assertion below
    reads a typed field instead of indexing YAML's untyped mapping."""
    block = frontmatter["promote"]
    assert isinstance(block, dict)
    assert isinstance(block.get("target"), str)
    assert isinstance(block.get("fact"), str)
    assert isinstance(block.get("source"), str)
    assert isinstance(block.get("relations"), list)
    return cast(_PromoteBlock, block)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _inbox_notes(paths: VaultPaths) -> list[Path]:
    d = paths.root / _INBOX
    return sorted(d.glob("*.md")) if d.is_dir() else []


_PERSON = """---
title: "{title}"
type: person
created: '2026-01-01'
updated: '2026-01-01'
{relations}---

# {title}
"""


def _person(title: str, *, relations: str = "") -> str:
    return _PERSON.format(title=title, relations=relations)


# ---------------------------------------------------------------------------
# propose_fact — the generic proposer
# ---------------------------------------------------------------------------

def test_propose_writes_note_matching_memory_fact_contract(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    result = propose_fact(
        paths,
        kind="wikilink-relation",
        title="related_to candidate: people/anna -> people/bob",
        target="people/anna",
        source="knowledge/people/anna",
        reason="Evidence: [[knowledge/people/bob]] appears in [[knowledge/people/anna]].",
        relations=[Relation(rel="related_to", target="people/bob", source="knowledge/people/anna")],
        now=NOW,
    )
    assert result.action == "proposed"
    notes = _inbox_notes(paths)
    assert len(notes) == 1
    assert notes[0].relative_to(paths.root).as_posix() == result.path

    text = notes[0].read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    assert fm["type"] == "memory_fact"
    assert fm["memory_status"] == "unconsolidated"
    assert fm["confirmations"] == 0
    assert fm["approved"] is False
    assert fm["author"] == "script:wikilink-relation"
    assert fm["written_via"] == "script"
    assert fm["created"] == "2026-06-12T09:30:00Z"

    promote = _promote(fm)
    assert promote["target"] == "people/anna"
    assert promote["fact"] == ""
    assert promote["source"] == "knowledge/people/anna"
    relations, problems = parse_relations({"relations": promote["relations"]})
    assert problems == []
    assert relations == [Relation(rel="related_to", target="people/bob", source="knowledge/people/anna")]

    assert "[[knowledge/people/bob]]" in body
    assert "[[knowledge/people/anna]]" in body


def test_propose_empty_relations_serializes_as_empty_list(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    propose_fact(
        paths, kind="merge", title="t", target="people/anna",
        source="knowledge/people/anna", reason="why", now=NOW,
    )
    text = _inbox_notes(paths)[0].read_text(encoding="utf-8")
    fm, _body = _split_frontmatter(text)
    assert _promote(fm)["relations"] == []


def _propose_anna_bob(paths: VaultPaths, *, now: datetime) -> ProposeResult:
    """The one proposal the idempotency tests re-issue verbatim. Written as
    explicit keyword arguments (not a **kwargs dict) so the call is checked
    against propose_fact's real signature."""
    return propose_fact(
        paths,
        kind="wikilink-relation",
        title="t",
        target="people/anna",
        source="knowledge/people/anna",
        reason="why",
        relations=[Relation(rel="related_to", target="people/bob")],
        now=now,
    )


def test_propose_is_idempotent_on_rerun(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    first = _propose_anna_bob(paths, now=NOW)
    assert first.action == "proposed"
    before = _inbox_notes(paths)[0].read_bytes()

    second = _propose_anna_bob(paths, now=datetime(2027, 1, 1, tzinfo=UTC))
    assert second.action == "exists"
    assert second.path == first.path
    assert len(_inbox_notes(paths)) == 1
    assert _inbox_notes(paths)[0].read_bytes() == before  # untouched, not re-stamped


def test_propose_never_overwrites_an_edited_existing_note(tmp_path: Path) -> None:
    """A human may hand-edit a proposal in the inbox (approve it, add a
    note). A rerun proposing the exact same content must not clobber that
    edit — the deterministic filename means "already proposed", not
    "safe to regenerate"."""
    paths = _vault(tmp_path)
    result = _propose_anna_bob(paths, now=NOW)
    note = paths.root / result.path
    edited = note.read_text(encoding="utf-8").replace("approved: false", "approved: true")
    note.write_text(edited, encoding="utf-8")

    rerun = _propose_anna_bob(paths, now=NOW)
    assert rerun.action == "exists"
    assert note.read_text(encoding="utf-8") == edited


def test_propose_different_content_gets_a_different_file(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    a = propose_fact(
        paths, kind="wikilink-relation", title="t", target="people/anna",
        source="knowledge/people/anna", reason="r", now=NOW,
        relations=[Relation(rel="related_to", target="people/bob")],
    )
    b = propose_fact(
        paths, kind="wikilink-relation", title="t", target="people/anna",
        source="knowledge/people/anna", reason="r", now=NOW,
        relations=[Relation(rel="related_to", target="people/carl")],
    )
    assert a.path != b.path
    assert len(_inbox_notes(paths)) == 2


def test_declared_identity_pins_the_filename_across_rewordings(
    tmp_path: Path,
) -> None:
    """A pass that GENERATES its ``fact``/``source`` prose declares what
    actually identifies its candidate. Editing the wording then re-proposes
    the same file — it does not mint a second note beside the one already
    waiting in the human's inbox (which is what a scheduled pass would do on
    the night after its prose was edited)."""
    paths = _vault(tmp_path)
    first = propose_fact(
        paths, kind="entity-duplicate-merge", title="candidate",
        target="people/dana-scott", source="people/danaexamplecom",
        reason="first wording", fact="dana@example.com appears to be the same",
        identity=("people/danaexamplecom", "people/dana-scott"), now=NOW,
    )
    second = propose_fact(
        paths, kind="entity-duplicate-merge", title="candidate (reworded)",
        target="people/dana-scott", source="knowledge/people/danaexamplecom",
        reason="second wording", fact="dana@example.com is the same entity",
        identity=("people/danaexamplecom", "people/dana-scott"), now=NOW,
    )

    assert first.action == "proposed"
    assert second.action == "exists"
    assert second.path == first.path
    assert len(_inbox_notes(paths)) == 1


def test_declared_identity_still_separates_different_candidates(
    tmp_path: Path,
) -> None:
    """Identity narrows the key; it does not collapse it. Two different
    merge pairs onto the same survivor are still two proposals."""
    paths = _vault(tmp_path)
    a = propose_fact(
        paths, kind="entity-duplicate-merge", title="candidate",
        target="people/dana-scott", source="knowledge/people/danaexamplecom",
        reason="r", identity=("people/danaexamplecom", "people/dana-scott"),
        now=NOW,
    )
    b = propose_fact(
        paths, kind="entity-duplicate-merge", title="candidate",
        target="people/dana-scott", source="knowledge/people/danaoldmailcom",
        reason="r", identity=("people/danaoldmailcom", "people/dana-scott"),
        now=NOW,
    )

    assert a.path != b.path
    assert len(_inbox_notes(paths)) == 2


def test_propose_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    result = propose_fact(
        paths, kind="wikilink-relation", title="t", target="people/anna",
        source="knowledge/people/anna", reason="why",
        relations=[Relation(rel="related_to", target="people/bob")],
        now=NOW, dry_run=True,
    )
    assert result.action == "proposed"
    assert not (paths.root / _INBOX).exists()  # not even the directory


def test_propose_quoting_round_trips_special_characters(tmp_path: Path) -> None:
    """Regression: a title containing a colon, or a fact containing a
    quote, must come back byte-identical through YAML — not merely
    Python-repr-escaped text that happens to look plausible."""
    paths = _vault(tmp_path)
    result = propose_fact(
        paths, kind="merge", title="candidate: people/a -> people/b",
        target="people/a", source="knowledge/people/a",
        reason="it's a \"test\" — quotes: colons; all sorts",
        fact="it's a fact, with a colon: right here",
        now=NOW,
    )
    text = (paths.root / result.path).read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    assert fm["title"] == "candidate: people/a -> people/b"
    assert _promote(fm)["fact"] == "it's a fact, with a colon: right here"
    assert body.strip() == 'it\'s a "test" — quotes: colons; all sorts'


# ---------------------------------------------------------------------------
# wikilinks — the deterministic pass built on propose_fact
# ---------------------------------------------------------------------------

def test_find_candidates_basic_link(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/bob]]\n")
    _write(paths, "knowledge/people/bob.md", _person("Bob"))
    candidates = find_candidates(paths)
    assert [(c.source, c.target) for c in candidates] == [("people/anna", "people/bob")]


def test_existing_typed_relation_suppresses_candidate(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/anna.md",
        _person("Anna", relations="relations:\n- rel: works_at\n  target: organisations/acme\n")
        + "\n[[knowledge/organisations/acme]]\n",
    )
    _write(paths, "knowledge/organisations/acme.md", _person("Acme"))
    assert find_candidates(paths) == []


def test_link_to_nonexistent_note_is_skipped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/ghost]]\n")
    assert find_candidates(paths) == []


def test_link_outside_entity_dirs_is_skipped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/research/some-topic]]\n")
    _write(paths, "knowledge/research/some-topic.md", "---\ntitle: X\ntype: research\n---\n\nbody\n")
    assert find_candidates(paths) == []


def test_source_outside_entity_dirs_is_not_scanned(tmp_path: Path) -> None:
    """The feedback-loop guard: a note under knowledge/assistant/ (e.g. an
    inbox fact note whose body wikilinks the very entities it's about)
    must never itself be treated as a candidate SOURCE, or the pass would
    propose relations about its own previous proposals forever."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    _write(
        paths, f"{_INBOX}/some-earlier-proposal.md",
        "---\ntitle: x\ntype: memory_fact\nmemory_status: unconsolidated\n"
        "confirmations: 0\napproved: false\n"
        "promote:\n  target: people/anna\n  relations: []\n  fact: ''\n  source: ''\n"
        "---\n\n[[knowledge/people/anna]]\n",
    )
    assert find_candidates(paths) == []


def test_self_link_is_skipped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/anna]]\n")
    assert find_candidates(paths) == []


def test_duplicate_link_forms_dedup_to_one_candidate(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/anna.md",
        _person("Anna") + "\n[[knowledge/people/bob]] and later [[people/bob|Bob]]\n",
    )
    _write(paths, "knowledge/people/bob.md", _person("Bob"))
    assert len(find_candidates(paths)) == 1


def test_short_folder_project_target_resolves_via_normalize_target(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[projects/server]]\n")
    _write(paths, "knowledge/projects/server/server.md", _person("Server project"))
    candidates = find_candidates(paths)
    assert [(c.source, c.target) for c in candidates] == [("people/anna", "projects/server/server")]


def test_propose_wikilink_relations_writes_a_related_to_candidate(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/bob]]\n")
    _write(paths, "knowledge/people/bob.md", _person("Bob"))

    results = propose_wikilink_relations(paths)
    assert [r.action for r in results] == ["proposed"]

    notes = _inbox_notes(paths)
    assert len(notes) == 1
    fm, _body = _split_frontmatter(notes[0].read_text(encoding="utf-8"))
    promote = _promote(fm)
    assert promote["target"] == "people/anna"
    assert promote["source"] == "knowledge/people/anna"
    relations, problems = parse_relations({"relations": promote["relations"]})
    assert problems == []
    assert relations == [Relation(rel="related_to", target="people/bob", source="knowledge/people/anna")]


def test_propose_wikilink_relations_rerun_is_a_noop(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/bob]]\n")
    _write(paths, "knowledge/people/bob.md", _person("Bob"))

    propose_wikilink_relations(paths)
    before = {p.name: p.read_bytes() for p in _inbox_notes(paths)}

    second = propose_wikilink_relations(paths)
    assert [r.action for r in second] == ["exists"]
    after = {p.name: p.read_bytes() for p in _inbox_notes(paths)}
    assert after == before


def test_propose_wikilink_relations_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/bob]]\n")
    _write(paths, "knowledge/people/bob.md", _person("Bob"))

    results = propose_wikilink_relations(paths, dry_run=True)
    assert [r.action for r in results] == ["proposed"]
    assert not (paths.root / _INBOX).exists()


def _note(title: str, type_: str, *, extra_fm: str = "", relations: str = "", body: str = "") -> str:
    return (
        f"---\ntitle: \"{title}\"\ntype: {type_}\ncreated: '2026-01-01'\n"
        f"updated: '2026-01-01'\n{extra_fm}{relations}---\n\n# {title}\n{body}"
    )


def test_wikilink_pass_skips_a_projects_own_notes_linking_its_overview(tmp_path: Path) -> None:
    """A log note under projects/x/ linking [[projects/x/x]] is navigation,
    not a relation; a link from there to ANOTHER project still counts — and
    is attributed to the PROJECT (C2), with the log note as provenance."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/agent/agent.md", _note("Agent", "project"))
    _write(paths, "knowledge/projects/brain/brain.md", _note("Brain", "project"))
    _write(
        paths, "knowledge/projects/agent/log/2026-07-27.md",
        _note("Agent log", "project",
              body="See [[knowledge/projects/agent/agent]] and [[knowledge/projects/brain/brain]].\n"),
    )
    candidates = find_candidates(paths)
    pairs = {(c.source, c.target) for c in candidates}
    assert ("projects/agent/agent", "projects/agent/agent") not in pairs
    assert pairs == {("projects/agent/agent", "projects/brain/brain")}
    assert candidates[0].source_note == "projects/agent/log/2026-07-27"


def test_wikilink_pass_redirects_a_superseded_target_to_its_survivor(tmp_path: Path) -> None:
    """A link to a merged duplicate proposes an edge to the survivor, and
    nothing at all once the survivor is already related."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna-kowalska.md", _note("Anna Kowalska", "person"))
    _write(
        paths, "knowledge/people/annakowalskaexamplecom.md",
        _note("anna.kowalska@example.com", "person",
              extra_fm="superseded_by: knowledge/people/anna-kowalska\n"),
    )
    meeting = "knowledge/meetings/2026/2026-07-02-call.md"
    link = "With [[knowledge/people/annakowalskaexamplecom]].\n"
    _write(paths, meeting, _note("Call", "meeting", body=link))
    assert [(c.source, c.target) for c in find_candidates(paths)] == [
        ("meetings/2026/2026-07-02-call", "people/anna-kowalska")
    ]
    _write(
        paths, meeting,
        _note("Call", "meeting", body=link,
              relations="relations:\n  - rel: attended\n    target: people/anna-kowalska\n"),
    )
    assert find_candidates(paths) == []


# ---------------------------------------------------------------------------
# C1 — "already proposed" is a vault-wide fact, not an inbox-directory one
# ---------------------------------------------------------------------------

def test_archived_proposal_is_never_reproposed(tmp_path: Path) -> None:
    """C1, the honest rejection path: a candidate the human leaves alone is
    digested and MOVED to knowledge/assistant/archive/ after stale_days.
    Before this fix the next run wrote it straight back into the inbox, so
    the same candidate came round forever and could not be declined."""
    paths = _vault(tmp_path)
    first = _propose_anna_bob(paths, now=NOW)
    assert first.action == "proposed"

    stats = consolidate(paths, logger=_LOG, as_of=date(2026, 8, 1), stale_days=30)
    assert stats.digested == 1
    assert _inbox_notes(paths) == []
    archived = sorted((paths.root / _ARCHIVE).rglob("*.md"))
    assert [a.name for a in archived] == [Path(first.path).name]

    rerun = _propose_anna_bob(paths, now=datetime(2026, 8, 2, tzinfo=UTC))
    # N5: the note is NOT in the inbox any more, so the result must not name
    # the inbox path a caller would log or open.
    assert rerun.action == "archived"
    assert rerun.path == str(archived[0].relative_to(paths.root))
    assert (paths.root / rerun.path).is_file()
    assert _inbox_notes(paths) == []


def test_hand_archived_proposal_with_a_collision_suffix_is_never_reproposed(
    tmp_path: Path,
) -> None:
    """Rejecting a candidate at once means moving it to the archive by hand.
    consolidate mints -2/-3 suffixes on a name collision there, so the
    suffixed form has to count as handled too."""
    paths = _vault(tmp_path)
    result = _propose_anna_bob(paths, now=NOW)
    note = paths.root / result.path
    archived = paths.root / _ARCHIVE / "2026-06" / f"{note.stem}-2.md"
    archived.parent.mkdir(parents=True, exist_ok=True)
    note.rename(archived)

    rerun = _propose_anna_bob(paths, now=NOW)
    assert rerun.action == "archived"
    assert rerun.path == f"{_ARCHIVE}/2026-06/{note.stem}-2.md"
    assert _inbox_notes(paths) == []


def test_an_unrelated_archived_proposal_does_not_suppress_this_one(tmp_path: Path) -> None:
    """The archive check must match this proposal's own filename, not any
    file whose name merely starts the same way."""
    paths = _vault(tmp_path)
    dry = propose_fact(
        paths, kind="wikilink-relation", title="t", target="people/anna",
        source="knowledge/people/anna", reason="why",
        relations=[Relation(rel="related_to", target="people/bob")],
        now=NOW, dry_run=True,
    )
    stem = Path(dry.path).stem
    archived = paths.root / _ARCHIVE / "2026-05" / f"{stem}-other.md"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text("---\ntitle: x\n---\n", encoding="utf-8")

    assert _propose_anna_bob(paths, now=NOW).action == "proposed"
    assert len(_inbox_notes(paths)) == 1


# ---------------------------------------------------------------------------
# C2 — only canonical entity nodes are candidate ends
# ---------------------------------------------------------------------------

def test_link_to_a_project_log_note_is_attributed_to_the_project(tmp_path: Path) -> None:
    """A dated log note is a diary entry, not a graph node: a link TO one
    proposes an edge to the project that owns it, never to the log note
    (which would write a relations: block into a diary entry)."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/kern/kern.md", _note("Kern", "project"))
    _write(paths, "knowledge/projects/kern/log/2026-06-30.md", _note("Kern log", "project"))
    _write(
        paths, "knowledge/people/anna.md",
        _person("Anna") + "\nSee [[knowledge/projects/kern/log/2026-06-30]].\n",
    )
    assert [(c.source, c.target) for c in find_candidates(paths)] == [
        ("people/anna", "projects/kern/kern")
    ]


def test_notes_that_are_not_entity_nodes_produce_no_candidates(tmp_path: Path) -> None:
    """projects/shared/ is the cross-project folder, not a project, and a
    concept note is no node at all — neither can be a candidate SOURCE."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    _write(
        paths, "knowledge/projects/shared/roadmap.md",
        _note("Roadmap", "note", body="With [[knowledge/people/anna]].\n"),
    )
    _write(
        paths, "knowledge/concepts/some-idea.md",
        _note("Some idea", "concept", body="With [[knowledge/people/anna]].\n"),
    )
    assert find_candidates(paths) == []


def test_project_note_without_an_overview_yields_no_candidate(tmp_path: Path) -> None:
    """No overview note means no addressable node: proposing here would
    write a promote.target consolidate can never resolve, leaving the note
    stuck in the inbox forever."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    _write(
        paths, "knowledge/projects/ghost/log/2026-07-01.md",
        _note("Ghost log", "project", body="With [[knowledge/people/anna]].\n"),
    )
    assert find_candidates(paths) == []


def test_one_candidate_per_edge_across_a_projects_notes(tmp_path: Path) -> None:
    """Two notes in one project folder linking the same person are one edge
    on the project, not two proposals."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    _write(paths, "knowledge/projects/kern/kern.md", _note("Kern", "project"))
    link = "With [[knowledge/people/anna]].\n"
    _write(paths, "knowledge/projects/kern/log/2026-06-30.md",
           _note("Log A", "project", body=link))
    _write(paths, "knowledge/projects/kern/log/2026-07-01.md",
           _note("Log B", "project", body=link))
    candidates = find_candidates(paths)
    # The pair is emitted in sorted order (N1); the project is still the end
    # the link was found on, and the first log note is still the provenance.
    assert [(c.source, c.target) for c in candidates] == [("people/anna", "projects/kern/kern")]
    assert candidates[0].found_on == "projects/kern/kern"
    assert candidates[0].source_note == "projects/kern/log/2026-06-30"


def test_a_projects_curated_note_is_attributed_to_the_project(tmp_path: Path) -> None:
    """A curated note beside a project's overview is a node of its own
    (canonical_entity_id says so, and relations point at such notes), but
    prose written there is a claim about the PROJECT: the candidate is
    attributed to the overview, with the curated note as provenance."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/brain/brain.md", _note("Brain", "project"))
    _write(paths, "knowledge/projects/kern/kern.md", _note("Kern", "project"))
    _write(
        paths, "knowledge/projects/brain/autonomous-audit.md",
        _note("Audit", "project", body="Compared with [[knowledge/projects/kern/kern]].\n"),
    )
    candidates = find_candidates(paths)
    assert [(c.source, c.target) for c in candidates] == [
        ("projects/brain/brain", "projects/kern/kern")
    ]
    assert candidates[0].source_note == "projects/brain/autonomous-audit"


# ---------------------------------------------------------------------------
# N1 — related_to is undirected: one pair is one candidate, not a mirrored two
# ---------------------------------------------------------------------------

def test_mirrored_links_are_one_candidate(tmp_path: Path) -> None:
    """A and B each linking the other is ONE undirected edge. Proposing it
    twice put 4 mirrored proposals into a live queue of 10 and, at scale,
    30 proposals for 15 edges — and the pass's own suppression rule already
    treats a relation declared either way round as covering the pair."""
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/projects/zeta/zeta.md",
        _note("Zeta", "project", body="Built on [[knowledge/projects/alpha/alpha]].\n"),
    )
    _write(
        paths, "knowledge/projects/alpha/alpha.md",
        _note("Alpha", "project", body="Used by [[knowledge/projects/zeta/zeta]].\n"),
    )
    candidates = find_candidates(paths)
    assert [(c.source, c.target) for c in candidates] == [
        ("projects/alpha/alpha", "projects/zeta/zeta")
    ]
    # Provenance is the first note seen in node-id order, which here is the
    # end the pair is NOT written on — so the body has to say so.
    assert candidates[0].found_on == "projects/alpha/alpha"
    assert candidates[0].source_note == "projects/alpha/alpha"

    assert [r.action for r in propose_wikilink_relations(paths)] == ["proposed"]
    assert len(_inbox_notes(paths)) == 1


def test_mirrored_links_keep_the_provenance_of_the_first_note_seen(
    tmp_path: Path,
) -> None:
    """The emitted end is the one that sorts first, but the evidence is the
    note the link was actually found in — which may belong to the other
    end. The body must describe that note honestly rather than claiming the
    proposal's own source note linked anything."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/alpha/alpha.md", _note("Alpha", "project"))
    _write(
        paths, "knowledge/projects/zeta/zeta.md",
        _note("Zeta", "project", body="Built on [[knowledge/projects/alpha/alpha]].\n"),
    )
    candidates = find_candidates(paths)
    assert [(c.source, c.target) for c in candidates] == [
        ("projects/alpha/alpha", "projects/zeta/zeta")
    ]
    assert candidates[0].found_on == "projects/zeta/zeta"

    propose_wikilink_relations(paths)
    fm, body = _split_frontmatter(_inbox_notes(paths)[0].read_text(encoding="utf-8"))
    promote = _promote(fm)
    assert promote["target"] == "projects/alpha/alpha"
    assert promote["source"] == "knowledge/projects/zeta/zeta"
    assert "[[knowledge/projects/zeta/zeta]] links [[knowledge/projects/alpha/alpha]]" in body


# ---------------------------------------------------------------------------
# M1 — an edge already declared in EITHER direction is not a candidate
# ---------------------------------------------------------------------------

def test_inverse_typed_relation_suppresses_candidate(tmp_path: Path) -> None:
    """The target already declares a relation back to the source, so the
    proposal's own claim ("no typed relation between them") would be false."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[knowledge/people/bob]]\n")
    _write(
        paths, "knowledge/people/bob.md",
        _person("Bob", relations="relations:\n- rel: collaborator_on\n  target: people/anna\n"),
    )
    assert find_candidates(paths) == []


def test_meeting_attendee_link_is_suppressed(tmp_path: Path) -> None:
    """A meeting's '- Attendee: [[…]]' bullet renders its own attendees:
    frontmatter — the edge is already declared, not discovered."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    _write(paths, "knowledge/people/bob.md", _person("Bob"))
    _write(
        paths, "knowledge/meetings/2026/2026-07-02-call.md",
        _note("Call", "meeting",
              extra_fm="attendees:\n- knowledge/people/anna\n",
              body="- Attendee: [[knowledge/people/anna]]\n"
                   "- Also: [[knowledge/people/bob]]\n"),
    )
    assert [(c.source, c.target) for c in find_candidates(paths)] == [
        ("meetings/2026/2026-07-02-call", "people/bob")
    ]


def test_short_form_stored_relation_suppresses_candidate(tmp_path: Path) -> None:
    """AGENTS.md tolerates the short folder form on input, so a relation
    stored as `projects/server` must suppress a `projects/server/server`
    candidate — the two only compare equal once both are normalised
    against the vault."""
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/anna.md",
        _person("Anna", relations="relations:\n- rel: works_at\n  target: projects/server\n")
        + "\n[[knowledge/projects/server/server]]\n",
    )
    _write(paths, "knowledge/projects/server/server.md", _note("Server", "project"))
    assert find_candidates(paths) == []


# ---------------------------------------------------------------------------
# minors — what counts as prose, and what the proposal body may link
# ---------------------------------------------------------------------------

def test_frontmatter_and_fenced_links_are_not_prose(tmp_path: Path) -> None:
    """A structural frontmatter key and a YAML example inside a code fence
    are not assertions about this vault, so neither yields a candidate."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/bob.md", _person("Bob"))
    _write(paths, "knowledge/people/carl.md", _person("Carl"))
    _write(
        paths, "knowledge/people/anna.md",
        _note("Anna", "person", extra_fm='colleague: "[[knowledge/people/bob]]"\n',
              body="```yaml\nrelations:\n- target: [[knowledge/people/carl]]\n```\n"),
    )
    assert find_candidates(paths) == []


def test_malformed_frontmatter_is_still_stripped_before_the_link_scan(
    tmp_path: Path,
) -> None:
    """N3: _split_frontmatter hands the WHOLE text back as the body when the
    YAML fails to parse, so a structural key in a broken fence would be read
    as a discovered prose link — on exactly the notes least worth trusting.
    The fence is stripped textually instead."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/bob.md", _person("Bob"))
    good = _note("Anna", "person", extra_fm='colleague: "[[knowledge/people/bob]]"\n')
    _write(paths, "knowledge/people/anna.md", good)
    assert find_candidates(paths) == []

    broken = good.replace("type: person", "topics: [a, b")
    _write(paths, "knowledge/people/anna.md", broken)
    assert find_candidates(paths) == []


def test_attendee_suppression_survives_malformed_frontmatter(tmp_path: Path) -> None:
    """N3, the other half: the attendees: mapping is read from the fenced
    block, so a meeting whose YAML is broken must not lose the suppression
    and start proposing its own generated Attendee bullets."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    meeting = _note(
        "Call", "meeting",
        extra_fm="attendees:\n- knowledge/people/anna\n",
        body="- Attendee: [[knowledge/people/anna]]\n",
    )
    _write(paths, "knowledge/meetings/2026/2026-07-02-call.md", meeting)
    assert find_candidates(paths) == []

    # Broken YAML: the attendees list is still there to be read, and the
    # bullet below the fence is still the frontmatter rendered back.
    _write(
        paths, "knowledge/meetings/2026/2026-07-02-call.md",
        meeting.replace("type: meeting", "topics: [a, b"),
    )
    assert find_candidates(paths) == []


def test_proposal_body_links_only_canonical_forms(tmp_path: Path) -> None:
    """The pass accepts the short folder form on input; echoing it back as a
    live link would mint a dangling/non-canonical sweep finding per
    proposal, so the body links the canonical note and quotes the raw form
    as inline code."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna") + "\n[[projects/server]]\n")
    _write(paths, "knowledge/projects/server/server.md", _note("Server", "project"))

    propose_wikilink_relations(paths)
    body = _split_frontmatter(_inbox_notes(paths)[0].read_text(encoding="utf-8"))[1]
    assert "[[projects/server]]" not in body
    assert "`projects/server`" in body
    assert "[[knowledge/projects/server/server]]" in body


def test_wikilink_proposal_records_the_note_the_link_came_from(tmp_path: Path) -> None:
    """promote.source and the relation's source: are the LOG note the link
    was found in; promote.target is the project it belongs to."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _person("Anna"))
    _write(paths, "knowledge/projects/kern/kern.md", _note("Kern", "project"))
    _write(paths, "knowledge/projects/kern/log/2026-06-30.md",
           _note("Kern log", "project", body="With [[knowledge/people/anna]].\n"))

    propose_wikilink_relations(paths)
    fm, body = _split_frontmatter(_inbox_notes(paths)[0].read_text(encoding="utf-8"))
    promote = _promote(fm)
    # The pair is written on the end that sorts first; the provenance is the
    # log note either way, and the body says which note the link came from.
    assert promote["target"] == "people/anna"
    assert promote["source"] == "knowledge/projects/kern/log/2026-06-30"
    assert "[[knowledge/projects/kern/log/2026-06-30]]" in body
    assert "[[knowledge/projects/kern/kern]]" in body
    relations, problems = parse_relations({"relations": promote["relations"]})
    assert problems == []
    assert relations == [Relation(
        rel="related_to", target="projects/kern/kern",
        source="knowledge/projects/kern/log/2026-06-30",
    )]
