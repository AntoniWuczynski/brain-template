"""Tests for ingest_lib.relations — typed, dated relationships between
knowledge entities: tolerant frontmatter parsing, node-id canonicalisation,
and the pure text-editing helpers the MCP entity tools build on."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import (
    RELATION_VOCAB,
    EntityInfo,
    Relation,
    append_fact_to_log,
    canonical_entity_id,
    entity_notes,
    live_entity_id,
    node_id_for_note,
    folder_overview_id,
    normalize_target,
    note_path_for_node,
    owning_entity_id,
    parse_relations,
    upsert_relation_in_text,
)


def _rels(fm: dict[str, object]) -> list[dict[str, object]]:
    """Narrow ``fm["relations"]`` for assertions: YAML round-trips hand back
    plain ``object`` values, and the asserts double as shape checks."""
    raw = fm["relations"]
    assert isinstance(raw, list)
    entries: list[dict[str, object]] = []
    for entry in raw:
        assert isinstance(entry, dict)
        entries.append({str(key): value for key, value in entry.items()})
    return entries


def _vault(tmp_path: Path) -> Path:
    for sub in (
        "knowledge/people", "knowledge/organisations", "knowledge/projects",
        "knowledge/meetings", "knowledge/index", "knowledge/concepts",
        "metadata", "inbox", "logs",
        "archive/raw", "archive/processed", "archive/failed",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# target / node-id canonicalisation
# ---------------------------------------------------------------------------

def test_normalize_target_collapses_all_tolerated_forms() -> None:
    for raw in (
        "people/x",
        "knowledge/people/x",
        "people/x.md",
        "knowledge/people/x.md",
        "[[knowledge/people/x]]",
        "[[knowledge/people/x|Display Name]]",
        "  people/x  ",
        "/knowledge/people/x",
    ):
        assert normalize_target(raw) == "people/x", raw


def test_node_id_round_trip() -> None:
    assert node_id_for_note("knowledge/people/anna-kowalska.md") == "people/anna-kowalska"
    assert note_path_for_node("people/anna-kowalska") == "knowledge/people/anna-kowalska.md"
    for node in ("people/anna", "organisations/acme", "meetings/2026/2026-06-12-kern-call"):
        assert node_id_for_note(note_path_for_node(node)) == node


def test_normalize_target_without_vault_root_is_pure_string_work(tmp_path: Path) -> None:
    """F9: the folder-overview resolution is opt-in. Callers that pass no
    ``vault_root`` (Relation.__post_init__, query filters, the MCP tools)
    keep the old behaviour even when the overview note exists on disk."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")

    assert normalize_target("projects/server") == "projects/server"
    assert normalize_target("[[knowledge/projects/server]]") == "projects/server"


def test_normalize_target_resolves_short_form_to_folder_overview(tmp_path: Path) -> None:
    """F9: a project is a folder, so ``projects/server`` names no note.
    With a vault_root it resolves to the overview that does exist."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")

    for raw in (
        "projects/server",
        "knowledge/projects/server",
        "[[knowledge/projects/server]]",
        "projects/server.md",
    ):
        assert normalize_target(raw, vault_root=root) == "projects/server/server", raw
    # Already canonical: returned untouched (the flat-note check matches it).
    assert normalize_target("projects/server/server", vault_root=root) == "projects/server/server"


def test_normalize_target_leaves_short_form_alone_when_flat_note_exists(tmp_path: Path) -> None:
    """A real note at the short path wins outright — the folder overview is
    only consulted when nothing sits at ``knowledge/<node_id>.md``."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/rex.md", "---\ntitle: Rex\n---\n")
    # A same-named folder overview also exists: the flat note still wins.
    _write(root, "knowledge/projects/rex/rex.md", "---\ntitle: Rex folder\n---\n")

    assert normalize_target("projects/rex", vault_root=root) == "projects/rex"


def test_normalize_target_unresolvable_target_is_returned_unchanged(tmp_path: Path) -> None:
    """Neither a flat note nor a folder overview: no guessing — the id comes
    back as written so the caller reports the miss."""
    root = _vault(tmp_path)
    (root / "knowledge/projects/ghost").mkdir(parents=True, exist_ok=True)
    _write(root, "knowledge/projects/ghost/notes.md", "---\ntitle: Notes\n---\n")

    assert normalize_target("projects/ghost", vault_root=root) == "projects/ghost"
    assert normalize_target("people/nobody", vault_root=root) == "people/nobody"


def test_normalize_target_never_resolves_a_traversal(tmp_path: Path) -> None:
    """A ``..`` target is not a valid node id, so resolution is skipped
    entirely — it must never be joined onto the vault root, and it must come
    back unchanged for ``is_valid_node_id`` downstream to reject."""
    root = _vault(tmp_path)
    _write(root, "AGENTS.md", "ORIGINAL\n")
    (root / "knowledge/AGENTS").mkdir(parents=True, exist_ok=True)
    _write(root, "knowledge/AGENTS/AGENTS.md", "decoy\n")

    for raw in ("../AGENTS", "projects/../../AGENTS", "people/./x"):
        assert normalize_target(raw, vault_root=root) == raw, raw
    # A single segment is a concept slug, not an entity: never expanded.
    assert normalize_target("server", vault_root=root) == "server"


def test_folder_overview_id_is_purely_textual() -> None:
    assert folder_overview_id("projects/server") == "projects/server/server"
    assert folder_overview_id("projects/a/b") == "projects/a/b/b"


def test_relation_dataclass_normalises_target_on_construction() -> None:
    r = Relation(rel="works_at", target="[[knowledge/organisations/acme]]")
    assert r.target == "organisations/acme"


# ---------------------------------------------------------------------------
# parse_relations tolerance matrix
# ---------------------------------------------------------------------------

def test_parse_relations_valid_entry() -> None:
    fm = {
        "relations": [
            {
                "rel": "works_at",
                "target": "knowledge/organisations/acme.md",
                "valid_from": "2025-03-01",
                "source": "knowledge/meetings/2026/2026-06-12-kern-call",
            }
        ]
    }
    relations, problems = parse_relations(fm)
    assert problems == []
    assert relations == [
        Relation(
            rel="works_at",
            target="organisations/acme",
            valid_from="2025-03-01",
            source="knowledge/meetings/2026/2026-06-12-kern-call",
        )
    ]


def test_parse_relations_absent_or_empty_is_clean() -> None:
    assert parse_relations({}) == ([], [])
    assert parse_relations({"relations": []}) == ([], [])
    assert parse_relations({"relations": None}) == ([], [])


def test_parse_relations_non_list_reports_problem() -> None:
    relations, problems = parse_relations({"relations": "works_at acme"})
    assert relations == []
    assert len(problems) == 1 and "expected a list" in problems[0]


def test_parse_relations_non_dict_entries_skipped_with_problem() -> None:
    relations, problems = parse_relations({"relations": ["works_at", 42]})
    assert relations == []
    assert len(problems) == 2
    assert all("expected a mapping" in p for p in problems)


def test_parse_relations_missing_rel_or_target_skipped() -> None:
    fm = {
        "relations": [
            {"target": "organisations/acme"},
            {"rel": "works_at"},
            {"rel": "works_at", "target": "organisations/acme"},
        ]
    }
    relations, problems = parse_relations(fm)
    assert [r.target for r in relations] == ["organisations/acme"]
    assert any("missing rel" in p for p in problems)
    assert any("missing target" in p for p in problems)


def test_parse_relations_unknown_rel_excluded_but_reported() -> None:
    assert "employed_by" not in RELATION_VOCAB
    fm = {
        "relations": [
            {"rel": "employed_by", "target": "organisations/acme"},
            {"rel": "works_at", "target": "organisations/acme"},
        ]
    }
    relations, problems = parse_relations(fm)
    assert [r.rel for r in relations] == ["works_at"]
    assert any("unknown rel 'employed_by'" in p for p in problems)


def test_parse_relations_unknown_key_reported_not_dropped() -> None:
    # F4: a typo'd key (valid_to for valid_until) must be reported and the
    # entry excluded — not silently parsed as if valid_until were empty,
    # which would make a closed span read as currently open.
    fm = {
        "relations": [
            {
                "rel": "attended",
                "target": "meetings/2026/x",
                "valid_from": "2026-08-19",
                "valid_to": "2026-08-21",
            },
            {"rel": "works_at", "target": "organisations/acme"},
        ]
    }
    relations, problems = parse_relations(fm)
    assert [r.rel for r in relations] == ["works_at"]  # bad entry excluded
    assert any(
        "unknown key 'valid_to' (allowed: rel, target, valid_from, "
        "valid_until, source)" in p
        for p in problems
    )


def test_parse_relations_multiple_unknown_keys_each_reported() -> None:
    fm = {"relations": [{"rel": "works_at", "target": "organisations/acme", "notes": "x", "who": "y"}]}
    relations, problems = parse_relations(fm)
    assert relations == []
    assert len(problems) == 2
    assert any("unknown key 'notes'" in p for p in problems)
    assert any("unknown key 'who'" in p for p in problems)


def test_parse_relations_yaml_dates_pass_through_as_iso_strings() -> None:
    # Unquoted YAML dates load as datetime.date — str() is the ISO form.
    fm = {
        "relations": [
            {"rel": "works_at", "target": "organisations/acme",
             "valid_from": date(2025, 3, 1), "valid_until": date(2026, 2, 28)}
        ]
    }
    relations, problems = parse_relations(fm)
    assert problems == []
    assert relations[0].valid_from == "2025-03-01"
    assert relations[0].valid_until == "2026-02-28"


# ---------------------------------------------------------------------------
# upsert_relation_in_text
# ---------------------------------------------------------------------------

PERSON_NOTE = """---
title: Anna Kowalska
type: person
aliases: [Ania]
---

# Anna Kowalska

Some body text that must survive byte-for-byte.

## Log

- 2026-06-01 — met at the Kern kickoff
"""


def _body(text: str) -> str:
    return _split_frontmatter(text)[1]


def test_upsert_adds_new_relation_and_preserves_body() -> None:
    new_text, action = upsert_relation_in_text(
        PERSON_NOTE,
        Relation(rel="works_at", target="organisations/acme", valid_from="2025-03-01"),
    )
    assert action == "added"
    assert _body(new_text) == _body(PERSON_NOTE)   # body byte-for-byte
    fm, _ = _split_frontmatter(new_text)
    assert fm["title"] == "Anna Kowalska"          # other keys survive
    assert fm["relations"] == [
        {"rel": "works_at", "target": "organisations/acme", "valid_from": "2025-03-01"}
    ]


def test_upsert_identical_open_relation_is_noop() -> None:
    relation = Relation(rel="works_at", target="organisations/acme", valid_from="2025-03-01")
    once, _ = upsert_relation_in_text(PERSON_NOTE, relation)
    twice, action = upsert_relation_in_text(once, relation)
    assert action == "noop"
    assert twice == once                            # text unchanged, byte-for-byte


def test_upsert_noop_matches_tolerant_target_forms() -> None:
    once, _ = upsert_relation_in_text(
        PERSON_NOTE, Relation(rel="works_at", target="organisations/acme")
    )
    _, action = upsert_relation_in_text(
        once, Relation(rel="works_at", target="[[knowledge/organisations/acme]]")
    )
    assert action == "noop"


def test_upsert_closes_open_relation_keeping_valid_from() -> None:
    once, _ = upsert_relation_in_text(
        PERSON_NOTE,
        Relation(rel="works_at", target="organisations/acme", valid_from="2025-03-01"),
    )
    closed, action = upsert_relation_in_text(
        once,
        Relation(rel="works_at", target="organisations/acme", valid_until="2026-02-28"),
    )
    assert action == "closed"
    assert _body(closed) == _body(PERSON_NOTE)
    fm, _ = _split_frontmatter(closed)
    assert fm["relations"] == [
        {
            "rel": "works_at",
            "target": "organisations/acme",
            "valid_from": "2025-03-01",      # kept
            "valid_until": "2026-02-28",     # set
        }
    ]


def test_upsert_after_close_appends_new_entry_history_intact() -> None:
    text = PERSON_NOTE
    text, _ = upsert_relation_in_text(
        text, Relation(rel="works_at", target="organisations/acme", valid_from="2025-03-01")
    )
    text, _ = upsert_relation_in_text(
        text, Relation(rel="works_at", target="organisations/acme", valid_until="2026-02-28")
    )
    text, action = upsert_relation_in_text(
        text, Relation(rel="works_at", target="organisations/acme", valid_from="2026-03-01")
    )
    assert action == "added"
    fm, _ = _split_frontmatter(text)
    rels = _rels(fm)
    assert len(rels) == 2                          # closed span + new open one
    assert rels[0]["valid_until"] == "2026-02-28"
    assert "valid_until" not in rels[1]


def test_upsert_repeated_close_without_valid_from_is_noop() -> None:
    # Open an entry carrying a valid_from, then close it the natural way:
    # a call with valid_until and NO valid_from ("this ended on Y").
    text, _ = upsert_relation_in_text(
        PERSON_NOTE,
        Relation(rel="works_at", target="organisations/acme", valid_from="2022-01-01"),
    )
    closed, action = upsert_relation_in_text(
        text,
        Relation(rel="works_at", target="organisations/acme", valid_until="2025-02-28"),
    )
    assert action == "closed"
    # Retrying the identical close must be a noop, not a bogus duplicate:
    # the entry's own valid_from ("2022-01-01") must match an empty
    # relation.valid_from.
    again, action_again = upsert_relation_in_text(
        closed,
        Relation(rel="works_at", target="organisations/acme", valid_until="2025-02-28"),
    )
    assert action_again == "noop"
    assert again == closed                          # text unchanged, byte-for-byte
    fm, _ = _split_frontmatter(again)
    rels = _rels(fm)
    assert len(rels) == 1                           # no duplicate appended
    assert rels[0]["valid_from"] == "2022-01-01"
    assert rels[0]["valid_until"] == "2025-02-28"


def test_upsert_close_with_explicit_different_valid_from_still_distinguishes() -> None:
    # Same valid_until but a DIFFERENT explicit valid_from is a distinct
    # historical span, so it must append rather than noop.
    text, _ = upsert_relation_in_text(
        PERSON_NOTE,
        Relation(rel="works_at", target="organisations/acme", valid_from="2022-01-01"),
    )
    text, _ = upsert_relation_in_text(
        text,
        Relation(rel="works_at", target="organisations/acme", valid_until="2025-02-28"),
    )
    text, action = upsert_relation_in_text(
        text,
        Relation(
            rel="works_at", target="organisations/acme",
            valid_from="2024-06-01", valid_until="2025-02-28",
        ),
    )
    assert action == "added"
    fm, _ = _split_frontmatter(text)
    assert len(_rels(fm)) == 2


_TWO_OPEN_SPANS = (
    "---\n"
    "title: P\n"
    "relations:\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "    valid_from: '2020-01-01'\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "    valid_from: '2024-01-01'\n"
    "---\n"
    "\n"
    "# P\n"
)


def test_upsert_close_names_the_right_open_span_by_valid_from() -> None:
    # F3: two open spans already exist for the same (rel, target) — legacy
    # bad data written before this fix, or a hand-edited note. A close
    # carrying valid_from must close the entry that STARTED on that date,
    # never entries[0] (the 2020 span) regardless of which was declared first.
    text, action = upsert_relation_in_text(
        _TWO_OPEN_SPANS,
        Relation(
            rel="works_at", target="organisations/acme",
            valid_from="2024-01-01", valid_until="2025-06-01",
        ),
    )
    assert action == "closed"
    fm, _ = _split_frontmatter(text)
    rels = _rels(fm)
    assert len(rels) == 2
    by_start = {r["valid_from"]: r for r in rels}
    assert "valid_until" not in by_start["2020-01-01"]        # untouched
    assert by_start["2024-01-01"]["valid_until"] == "2025-06-01"  # the one named


def test_upsert_adding_new_open_span_supersedes_prior_open_one() -> None:
    # F3: adding a second open entry for the same (rel, target) must never
    # leave two open spans — the prior open entry is superseded (its
    # valid_until set to the day before the new entry's valid_from).
    text = "---\ntitle: P\n---\n\n# P\n"
    text, action = upsert_relation_in_text(
        text, Relation(rel="works_at", target="organisations/acme", valid_from="2020-01-01")
    )
    assert action == "added"
    text, action = upsert_relation_in_text(
        text, Relation(rel="works_at", target="organisations/acme", valid_from="2024-01-01")
    )
    assert action == "added"
    fm, _ = _split_frontmatter(text)
    rels = _rels(fm)
    assert len(rels) == 2
    open_spans = [r for r in rels if "valid_until" not in r]
    assert len(open_spans) == 1                                # never two opens
    assert open_spans[0]["valid_from"] == "2024-01-01"
    by_start = {r["valid_from"]: r for r in rels}
    assert by_start["2020-01-01"]["valid_until"] == "2023-12-31"  # day before

    # Closing the still-open (2024) span afterwards must name IT, not 2020.
    text, action = upsert_relation_in_text(
        text,
        Relation(
            rel="works_at", target="organisations/acme",
            valid_from="2024-01-01", valid_until="2025-06-01",
        ),
    )
    assert action == "closed"
    fm, _ = _split_frontmatter(text)
    rels = _rels(fm)
    assert not any("valid_until" not in r for r in rels)        # no opens left
    by_start = {r["valid_from"]: r for r in rels}
    assert by_start["2020-01-01"]["valid_until"] == "2023-12-31"  # untouched
    assert by_start["2024-01-01"]["valid_until"] == "2025-06-01"


def test_upsert_refuses_fenced_frontmatter_that_fails_to_parse() -> None:
    # F2: same class as AUD-001 but for the relations writer specifically —
    # a fence IS present but doesn't parse as a YAML mapping (here: a
    # tab-indented value, invalid YAML). The old behaviour prepended a
    # SECOND fence over it, burying every existing key. Refuse instead.
    bad = "---\ntitle: X\n\trelations: []\n---\n\n# X\n"
    with pytest.raises(ValueError, match="does not parse as a YAML mapping"):
        upsert_relation_in_text(bad, Relation(rel="works_at", target="organisations/acme"))


def test_upsert_genuinely_empty_frontmatter_fence_still_works() -> None:
    # Positive control: a fence that parses to an empty mapping (not the
    # "identity fallback" F2 targets) must NOT be misclassified as broken.
    empty_fence = "---\n---\n\n# X\n"
    new_text, action = upsert_relation_in_text(
        empty_fence, Relation(rel="works_at", target="organisations/acme")
    )
    assert action == "added"
    fm, body = _split_frontmatter(new_text)
    assert body == "\n# X\n"
    assert _rels(fm)[0]["rel"] == "works_at"


def test_upsert_on_note_without_frontmatter_creates_fence() -> None:
    plain = "# Anna\n\nJust a body.\n"
    new_text, action = upsert_relation_in_text(
        plain, Relation(rel="met_at", target="meetings/2026/2026-06-12-kern-call")
    )
    assert action == "added"
    assert new_text.startswith("---\n")
    assert new_text.endswith(plain)                 # body preserved byte-for-byte
    fm, body = _split_frontmatter(new_text)
    assert body == plain
    assert _rels(fm)[0]["rel"] == "met_at"


# ---------------------------------------------------------------------------
# append_fact_to_log
# ---------------------------------------------------------------------------

def test_append_fact_to_existing_log_section_lands_last() -> None:
    result = append_fact_to_log(PERSON_NOTE, "2026-06-12 — joined ACME")
    assert "- 2026-06-01 — met at the Kern kickoff\n- 2026-06-12 — joined ACME" in result


def test_append_fact_inserts_before_following_section() -> None:
    note = "# T\n\n## Log\n\n- old\n\n## Open questions\n\n- q1\n"
    result = append_fact_to_log(note, "new fact")
    assert "- old\n- new fact" in result
    assert result.index("- new fact") < result.index("## Open questions")
    assert result.endswith("## Open questions\n\n- q1\n")


def test_append_fact_creates_log_section_at_end_when_absent() -> None:
    note = "---\ntitle: x\n---\n\n# X\n\nBody.\n"
    result = append_fact_to_log(note, "first fact")
    assert result.endswith("Body.\n\n## Log\n\n- first fact\n")


def test_append_fact_to_empty_log_section() -> None:
    note = "# T\n\n## Log\n"
    result = append_fact_to_log(note, "only fact")
    assert "## Log\n\n- only fact" in result


def test_append_fact_idempotent_on_exact_duplicate() -> None:
    # Crash-/retry-safety: re-applying the EXACT same fact line is a no-op,
    # so a consolidation rerun or a retried append never double-applies it.
    once = append_fact_to_log(PERSON_NOTE, "2026-06-12 — joined ACME")
    twice = append_fact_to_log(once, "2026-06-12 — joined ACME")
    assert twice == once                            # text unchanged
    assert once.count("- 2026-06-12 — joined ACME") == 1


def test_append_fact_idempotent_across_crlf_line_endings() -> None:
    # F137: a note that picked up CRLF endings (Windows editor/sync tool)
    # must still dedup, or the crash/retry-safety silently fails.
    crlf = "# T\r\n\r\n## Log\r\n\r\n- 2026-06-12 — joined ACME\r\n"
    result = append_fact_to_log(crlf, "2026-06-12 — joined ACME")
    assert result == crlf                           # no second bullet added


def test_append_fact_distinct_facts_both_append() -> None:
    # Different date/text/source remain distinct facts and both land.
    once = append_fact_to_log(PERSON_NOTE, "2026-06-12 — joined ACME")
    twice = append_fact_to_log(once, "2026-06-13 — shipped the index")
    assert twice != once
    assert "- 2026-06-12 — joined ACME" in twice
    assert "- 2026-06-13 — shipped the index" in twice


# ---------------------------------------------------------------------------
# entity_notes
# ---------------------------------------------------------------------------

PERSON_WITH_RELATIONS = """---
title: Anna Kowalska
type: person
aliases: [Ania]
updated: "2026-06-12T10:00:00Z"
relations:
  - rel: works_at
    target: knowledge/organisations/acme
    valid_from: "2025-03-01"
  - rel: invented_rel
    target: organisations/acme
---

# Anna Kowalska
"""


def test_entity_notes_builds_nodes_with_parsed_relations(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/people/anna-kowalska.md", PERSON_WITH_RELATIONS)
    _write(root, "knowledge/organisations/acme.md", "---\ntitle: ACME\ntype: organisation\n---\n\n# ACME\n")
    _write(
        root,
        "knowledge/meetings/2026/2026-06-12-kern-call.md",
        "Meeting notes without frontmatter.\n",
    )

    entities = entity_notes(paths_for_root(root))

    assert set(entities) == {
        "people/anna-kowalska",
        "organisations/acme",
        "meetings/2026/2026-06-12-kern-call",
    }
    anna = entities["people/anna-kowalska"]
    assert anna.rel_path == "knowledge/people/anna-kowalska.md"
    assert anna.title == "Anna Kowalska"
    assert anna.type == "person"
    assert anna.aliases == ("Ania",)
    assert anna.updated == "2026-06-12T10:00:00Z"
    # The unknown-vocab entry is excluded; the valid one is normalised.
    assert anna.relations == (
        Relation(rel="works_at", target="organisations/acme", valid_from="2025-03-01"),
    )
    # Title falls back to the filename stem when frontmatter is absent.
    assert entities["meetings/2026/2026-06-12-kern-call"].title == "2026-06-12-kern-call"


def test_entity_notes_deterministic_ordering_and_skips_generated_dirs(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/people/zed.md", "zed\n")
    _write(root, "knowledge/people/abe.md", "abe\n")
    _write(root, "knowledge/index/uni/lec.md", "generated\n")
    _write(root, "knowledge/concepts/prng.md", "generated\n")
    _write(root, "knowledge/people/empty.md", "   \n")

    paths = paths_for_root(root)
    first = list(entity_notes(paths))
    second = list(entity_notes(paths))

    assert first == ["people/abe", "people/zed"]   # sorted node ids, generated dirs out
    assert first == second


def test_entity_notes_excludes_assistant_history(tmp_path: Path) -> None:
    # F136: promoted-fact archive and swept digests are the audit trail, not
    # live entities — they must not appear as graph nodes (else vault_related
    # could resolve to an archived fact or a "Memory digest" note).
    root = _vault(tmp_path)
    _write(root, "knowledge/people/live.md", "---\ntitle: Live\n---\n")
    _write(root, "knowledge/assistant/archive/2026-06/fact-anna.md",
           "---\ntitle: Anna fact\n---\n")
    _write(root, "knowledge/assistant/digests/2026-06.md",
           "---\ntitle: Memory digest 2026-06\n---\n")

    entities = entity_notes(paths_for_root(root))

    assert "people/live" in entities
    assert not any(n.startswith("assistant/archive") for n in entities)
    assert not any(n.startswith("assistant/digests") for n in entities)


# ------------------------------------------------------------------
# query_relations (P3)

_ANNA_HISTORY = (
    "---\n"
    "title: Anna\n"
    "relations:\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "    valid_from: \"2025-01-01\"\n"
    "    valid_until: \"2026-01-01\"\n"        # ended
    "  - rel: works_at\n"
    "    target: organisations/initech\n"
    "    valid_from: \"2026-01-01\"\n"          # current (open)
    "    source: knowledge/meetings/2026/2026-01-05-x\n"
    "---\n"
)


def _history_vault(tmp_path: Path) -> VaultPaths:
    root = _vault(tmp_path)
    _write(root, "knowledge/people/anna.md", _ANNA_HISTORY)
    _write(root, "knowledge/organisations/acme.md", "---\ntitle: Acme\n---\n")
    _write(root, "knowledge/organisations/initech.md", "---\ntitle: Initech\n---\n")
    return paths_for_root(root)


def test_query_relations_open_only_by_default(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    hits = query_relations(paths)
    assert [(h.rel, h.target) for h in hits] == [("works_at", "organisations/initech")]


def test_query_relations_include_closed(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    hits = query_relations(paths, include_closed=True)
    targets = {h.target for h in hits}
    assert targets == {"organisations/acme", "organisations/initech"}


def test_query_relations_as_of_selects_historical_interval(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    # Mid-2025 -> Anna was at Acme, not Initech.
    hits = query_relations(paths, as_of="2025-06-01")
    assert [(h.target) for h in hits] == ["organisations/acme"]


def test_query_relations_reverse_lookup_by_target(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    # Who works at initech (open)? tolerant of the knowledge/ + .md forms.
    hits = query_relations(paths, target="knowledge/organisations/initech.md")
    assert [h.entity for h in hits] == ["people/anna"]
    assert hits[0].source == "knowledge/meetings/2026/2026-01-05-x"


# ---------------------------------------------------------------------------
# AUD-028 / AUD-029 / AUD-030 / AUD-035 / AUD-096
# ---------------------------------------------------------------------------

def test_upsert_leaves_every_other_frontmatter_key_byte_for_byte() -> None:
    # AUD-028: the old whole-fence yaml.safe_dump round-trip rewrote every
    # OTHER key on a relation write — `created: 2026-06-09T09:50:42Z` came
    # back as `2026-06-09 09:50:42+00:00`, flow lists exploded to block
    # style, quotes normalised and a long title folded onto a continuation
    # line. Only the relations block may change.
    note = (
        "---\n"
        "title: A very long title that goes on and on and on and really must "
        "not be folded onto a continuation line by the dumper\n"
        "created: 2026-06-09T09:50:42Z\n"
        "updated: 2026-06-09T09:50:42Z\n"
        "topics: [alpha, beta]\n"
        'type: "person"\n'
        "# a hand-written comment\n"
        "---\n"
        "# Body\n"
    )
    new_text, action = upsert_relation_in_text(
        note, Relation(rel="works_at", target="organisations/acme")
    )
    assert action == "added"
    for line in note.splitlines():
        if line != "---":
            assert line in new_text.splitlines()
    assert "created: 2026-06-09T09:50:42Z" in new_text
    assert "topics: [alpha, beta]" in new_text
    assert '# a hand-written comment' in new_text
    assert _rels(_split_frontmatter(new_text)[0])[0]["target"] == "organisations/acme"
    assert _body(new_text) == "# Body\n"


def test_upsert_rewrites_only_the_relations_block_of_an_existing_list() -> None:
    # The splice must replace the whole existing block (key line plus its
    # indented entries), never append a second `relations:` key.
    note = (
        "---\n"
        "title: Anna\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        "    valid_from: \"2025-01-01\"\n"
        "aliases: [Ania]\n"
        "---\n"
        "# Anna\n"
    )
    new_text, action = upsert_relation_in_text(
        note, Relation(rel="member_of", target="organisations/initech")
    )
    assert action == "added"
    assert new_text.count("relations:") == 1
    assert "title: Anna\n" in new_text
    assert "aliases: [Ania]\n" in new_text
    entries = _rels(_split_frontmatter(new_text)[0])
    assert [(e["rel"], e["target"]) for e in entries] == [
        ("works_at", "organisations/acme"),
        ("member_of", "organisations/initech"),
    ]


def test_upsert_does_not_fold_a_long_value_in_a_preserved_entry() -> None:
    # width must be unbounded: an existing malformed entry is kept verbatim
    # (supersede, never delete), and folding its long prose value onto a
    # continuation line churns the file for no gain.
    prose = (
        "closed when Anna left, per the handover note of the twelfth of June, "
        "which nobody has since rewritten as a date"
    )
    note = (
        "---\n"
        "title: Anna\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        f"    valid_until: {prose}\n"
        "---\n"
        "# Anna\n"
    )
    new_text, _action = upsert_relation_in_text(
        note, Relation(rel="member_of", target="organisations/initech")
    )
    assert f"  valid_until: {prose}\n" in new_text


def test_append_fact_never_lands_inside_a_fenced_code_block() -> None:
    # AUD-029: a '#' comment inside a ``` fence ended the section, so the
    # bullet was inserted into the code block ahead of the real last fact.
    note = (
        "## Log\n"
        "\n"
        "- 2026-01-01 — a\n"
        "```sh\n"
        "# comment\n"
        "```\n"
        "- 2026-01-02 — b\n"
    )
    result = append_fact_to_log(note, "2026-01-03 — c")
    assert result == note + "- 2026-01-03 — c\n"


def test_append_fact_ignores_a_log_heading_inside_a_fence() -> None:
    note = (
        "# T\n"
        "\n"
        "~~~md\n"
        "## Log\n"
        "~~~\n"
        "\n"
        "## Log\n"
        "\n"
        "- 2026-01-01 — a\n"
    )
    result = append_fact_to_log(note, "2026-01-02 — b")
    assert result.endswith("- 2026-01-01 — a\n- 2026-01-02 — b\n")
    assert result.count("## Log") == 2


def test_append_fact_replaces_the_empty_placeholder() -> None:
    # The templates ship `_(empty)_`; the first real fact takes its place
    # rather than sitting under it (a dated section is not empty).
    note = "## Log\n\n<!-- guidance -->\n\n_(empty)_\n"
    result = append_fact_to_log(note, "2026-01-03 — c")
    assert "_(empty)_" not in result
    assert result == "## Log\n\n<!-- guidance -->\n\n- 2026-01-03 — c\n"


def test_append_fact_keeps_a_placeholder_in_another_section() -> None:
    note = "## Notes\n\n_(empty)_\n\n## Log\n\n_(empty)_\n"
    result = append_fact_to_log(note, "2026-01-03 — c")
    assert result == "## Notes\n\n_(empty)_\n\n## Log\n\n- 2026-01-03 — c\n"


def test_append_fact_second_fact_leaves_the_first_alone() -> None:
    once = append_fact_to_log("## Log\n\n_(empty)_\n", "2026-01-03 — c")
    twice = append_fact_to_log(once, "2026-01-04 — d")
    assert twice == "## Log\n\n- 2026-01-03 — c\n- 2026-01-04 — d\n"


def _bad_date_vault(tmp_path: Path, valid_from: str) -> VaultPaths:
    root = _vault(tmp_path)
    _write(
        root, "knowledge/people/anna.md",
        "---\ntitle: Anna\nrelations:\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        f"    valid_from: {valid_from}\n---\n",
    )
    _write(root, "knowledge/organisations/acme.md", "---\ntitle: Acme\n---\n")
    return paths_for_root(root)


def test_query_relations_as_of_handles_a_yaml_datetime_bound(tmp_path: Path) -> None:
    # AUD-030: an unquoted `valid_from: 2025-03-01T00:00:00Z` loads as a
    # datetime whose str() is '2025-03-01 00:00:00+00:00'; comparing that
    # as raw text made it sort AFTER '2025-03-01' and the relation vanished
    # from an as_of query on its own start day.
    from ingest_lib.relations import query_relations
    paths = _bad_date_vault(tmp_path, "2025-03-01T00:00:00Z")
    hits = query_relations(paths, as_of="2025-03-01")
    assert [h.target for h in hits] == ["organisations/acme"]
    assert query_relations(paths, as_of="2025-02-28") == []


def test_query_relations_as_of_excludes_an_unparseable_bound(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _bad_date_vault(tmp_path, '"sometime in spring"')
    assert query_relations(paths, as_of="2025-03-01") == []
    # Without as_of the relation is still an open edge, unchanged.
    assert [h.target for h in query_relations(paths)] == ["organisations/acme"]


def test_query_relations_rejects_a_non_date_as_of(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _bad_date_vault(tmp_path, '"2025-03-01"')
    with pytest.raises(ValueError, match="as_of must be YYYY-MM-DD"):
        query_relations(paths, as_of="last spring")


@pytest.mark.parametrize(
    "target",
    ["acme", "assistant/PROFILE", "people/Łukasz", "people/../../archive/x"],
)
def test_parse_relations_rejects_targets_the_write_path_rejects(target: str) -> None:
    # AUD-035: node-id invariants were enforced on write (entity_upsert_relation
    # calls is_valid_node_id) but not on read, so a hand-edited note could put
    # a node the write path can never address into the graph.
    relations, problems = parse_relations(
        {"relations": [{"rel": "related_to", "target": target}]}
    )
    assert relations == []
    assert len(problems) == 1 and "invalid target node id" in problems[0]


def test_normalize_target_drops_a_heading_fragment() -> None:
    # A '#…' fragment names a heading inside a note, not a node.
    assert normalize_target("[[knowledge/people/anna.md#Log]]") == "people/anna"
    relations, problems = parse_relations(
        {"relations": [{"rel": "related_to", "target": "people/anna#Log"}]}
    )
    assert problems == []
    assert relations[0].target == "people/anna"


def test_agents_md_is_the_relation_vocabulary_source() -> None:
    # AUD-096: the vocabulary is hand-copied into several prose files with
    # no import. AGENTS.md is the documented source; this pins the code to it.
    agents_md = Path(__file__).resolve().parent.parent / "AGENTS.md"
    block = agents_md.read_text(encoding="utf-8").split("<!-- relation-vocab -->")[1]
    documented = set(re.findall(r"`([a-z_]+)`", block.split("<!-- /relation-vocab -->")[0]))
    assert documented == set(RELATION_VOCAB)


# ---------------------------------------------------------------------------
# canonical_entity_id / live_entity_id — the one definition of node identity
# ---------------------------------------------------------------------------

def test_canonical_entity_id_flat_entity_notes(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    assert canonical_entity_id(
        "knowledge/people/anna-kowalska.md", vault_root=root
    ) == "people/anna-kowalska"
    assert canonical_entity_id(
        "knowledge/organisations/acme.md", vault_root=root
    ) == "organisations/acme"
    # Negative: a note nested BELOW the flat folder is not a node of its own.
    assert canonical_entity_id(
        "knowledge/people/old/anna-kowalska.md", vault_root=root
    ) is None


def test_canonical_entity_id_meetings_need_a_year_folder(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    assert canonical_entity_id(
        "knowledge/meetings/2026/2026-06-12-kern-call.md", vault_root=root
    ) == "meetings/2026/2026-06-12-kern-call"
    # Negative: the year folder is part of the layout, not decoration.
    assert canonical_entity_id(
        "knowledge/meetings/kern-call.md", vault_root=root
    ) is None
    assert canonical_entity_id(
        "knowledge/meetings/drafts/kern-call.md", vault_root=root
    ) is None


def test_canonical_entity_id_project_overview_and_its_notes(tmp_path: Path) -> None:
    """A project's dated log entry belongs to the project's overview — it is
    not a graph node of its own (the bug that made dated log notes
    entities). A CURATED note beside the overview is the opposite case:
    AGENTS.md says it keeps its own path, and 36 of the live vault's stored
    relation targets are exactly such notes, so folding it into the project
    makes every consumer name the wrong node."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")
    _write(root, "knowledge/projects/server/log/2026-09-03.md", "# log\n")
    _write(root, "knowledge/projects/server/stack.md", "# stack\n")

    assert canonical_entity_id(
        "knowledge/projects/server/server.md", vault_root=root
    ) == "projects/server/server"
    assert canonical_entity_id(
        "knowledge/projects/server/log/2026-09-03.md", vault_root=root
    ) == "projects/server/server"
    assert canonical_entity_id(
        "knowledge/projects/server/stack.md", vault_root=root
    ) == "projects/server/stack"
    # Deeper than any documented shape, and not a log entry: no node.
    _write(root, "knowledge/projects/server/notes/deep/x.md", "# x\n")
    assert canonical_entity_id(
        "knowledge/projects/server/notes/deep/x.md", vault_root=root
    ) is None


def test_canonical_entity_id_flat_project_note(tmp_path: Path) -> None:
    """AGENTS.md tolerates a flat knowledge/projects/<slug>.md and one is
    live, so it is a node; the SAME two-segment id with a folder behind it
    instead is the short form normalize_target rewrites — no note, no node."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/zymbly-role.md", "---\ntitle: Zymbly\n---\n")
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")

    assert canonical_entity_id(
        "knowledge/projects/zymbly-role.md", vault_root=root
    ) == "projects/zymbly-role"
    assert canonical_entity_id("knowledge/projects/server.md", vault_root=root) is None


def test_owning_entity_id_attributes_a_projects_own_notes_to_the_project(
    tmp_path: Path,
) -> None:
    """The other identity question: which entity does prose written in this
    note belong to. A project's curated note is its own node but a document
    ABOUT the project, so its content is attributed to the overview;
    everything else owns what it says."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")
    _write(root, "knowledge/projects/server/stack.md", "# stack\n")
    _write(root, "knowledge/projects/server/log/2026-09-03.md", "# log\n")
    _write(root, "knowledge/people/anna-kowalska.md", "---\ntitle: Anna\n---\n")

    for rel in (
        "knowledge/projects/server/server.md",
        "knowledge/projects/server/stack.md",
        "knowledge/projects/server/log/2026-09-03.md",
    ):
        assert owning_entity_id(rel, vault_root=root) == "projects/server/server", rel
    assert owning_entity_id(
        "knowledge/people/anna-kowalska.md", vault_root=root
    ) == "people/anna-kowalska"
    assert owning_entity_id("knowledge/concepts/graphs.md", vault_root=root) is None
    # No overview to be attributed to: the note keeps its own content.
    _write(root, "knowledge/projects/orphan/stack.md", "# stack\n")
    assert owning_entity_id(
        "knowledge/projects/orphan/stack.md", vault_root=root
    ) == "projects/orphan/stack"


def test_canonical_entity_id_project_folder_without_an_overview(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/orphan/log/2026-09-03.md", "# log\n")
    assert canonical_entity_id(
        "knowledge/projects/orphan/log/2026-09-03.md", vault_root=root
    ) is None


def test_canonical_entity_id_shared_folder_is_not_a_project(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/shared/shared.md", "---\ntitle: Shared\n---\n")
    _write(root, "knowledge/projects/shared/hiring.md", "# hiring\n")
    assert canonical_entity_id(
        "knowledge/projects/shared/hiring.md", vault_root=root
    ) is None
    assert canonical_entity_id(
        "knowledge/projects/shared/shared.md", vault_root=root
    ) is None


def test_canonical_entity_id_non_entity_areas(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    for rel in (
        "knowledge/concepts/graphs.md",
        "knowledge/index/people.md",
        "knowledge/notes/dreams/questions.md",
        "knowledge/assistant/inbox/2026-09-03-fact.md",
        "knowledge/university/ucl.md",
        "knowledge/research/idea.md",
        "archive/processed/paper.md",
        "knowledge/projects/server.md",   # flat form with no note behind it
    ):
        assert canonical_entity_id(rel, vault_root=root) is None, rel


def _entity(node_id: str, superseded_by: str | None = None) -> EntityInfo:
    return EntityInfo(
        node_id=node_id,
        rel_path=f"knowledge/{node_id}.md",
        title=node_id,
        type="",
        aliases=(),
        relations=(),
        updated="",
        superseded_by=superseded_by,
    )


def test_entity_notes_reads_superseded_by(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(
        root, "knowledge/people/annakowalskaexamplecom.md",
        "---\ntitle: anna@example.com\n"
        "superseded_by: knowledge/people/anna-kowalska\n---\n",
    )
    _write(root, "knowledge/people/anna-kowalska.md", "---\ntitle: Anna\n---\n")

    entities = entity_notes(paths_for_root(root))
    dup = entities["people/annakowalskaexamplecom"]
    assert dup.superseded_by == "people/anna-kowalska"
    assert entities["people/anna-kowalska"].superseded_by is None


def test_live_entity_id_follows_a_chain_and_is_cycle_safe() -> None:
    entities = {
        "people/a": _entity("people/a", "people/b"),
        "people/b": _entity("people/b", "people/c"),
        "people/c": _entity("people/c"),
    }
    assert live_entity_id("people/a", entities) == "people/c"
    assert live_entity_id("people/c", entities) == "people/c"
    # Not an entity at all.
    assert live_entity_id("people/nobody", entities) is None
    # A pointer at a node that isn't an entity stops at the last live node.
    dangling = {"people/a": _entity("people/a", "people/gone")}
    assert live_entity_id("people/a", dangling) == "people/a"


def test_live_entity_id_never_returns_a_dangling_successor() -> None:
    """A superseded_by pointing at a note that does not exist (or that
    parsing dropped as an invalid id) must resolve to the last node that
    really is an entity — handing back the dangling id would let a merged
    duplicate grow edges on a target nothing can resolve."""
    entities = {
        "people/a": _entity("people/a", "people/b"),
        "people/b": _entity("people/b", "people/nowhere"),
    }
    for start in ("people/a", "people/b"):
        live = live_entity_id(start, entities)
        assert live == "people/b"
        assert live in entities


def test_live_entity_id_survives_a_superseded_cycle() -> None:
    entities = {
        "people/a": _entity("people/a", "people/b"),
        "people/b": _entity("people/b", "people/a"),
    }
    assert live_entity_id("people/a", entities) in {"people/a", "people/b"}
    assert live_entity_id("people/b", entities) in {"people/a", "people/b"}
