"""Tests for ingest_lib.relations — typed, dated relationships between
knowledge entities: tolerant frontmatter parsing, node-id canonicalisation,
and the pure text-editing helpers the MCP entity tools build on."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from ingest_lib.config import paths_for_root
from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import (
    RELATION_VOCAB,
    Relation,
    append_fact_to_log,
    entity_notes,
    node_id_for_note,
    normalize_target,
    note_path_for_node,
    parse_relations,
    upsert_relation_in_text,
)


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

def test_normalize_target_collapses_all_tolerated_forms():
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


def test_node_id_round_trip():
    assert node_id_for_note("knowledge/people/anna-kowalska.md") == "people/anna-kowalska"
    assert note_path_for_node("people/anna-kowalska") == "knowledge/people/anna-kowalska.md"
    for node in ("people/anna", "organisations/acme", "meetings/2026/2026-06-12-kern-call"):
        assert node_id_for_note(note_path_for_node(node)) == node


def test_relation_dataclass_normalises_target_on_construction():
    r = Relation(rel="works_at", target="[[knowledge/organisations/acme]]")
    assert r.target == "organisations/acme"


# ---------------------------------------------------------------------------
# parse_relations tolerance matrix
# ---------------------------------------------------------------------------

def test_parse_relations_valid_entry():
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


def test_parse_relations_absent_or_empty_is_clean():
    assert parse_relations({}) == ([], [])
    assert parse_relations({"relations": []}) == ([], [])
    assert parse_relations({"relations": None}) == ([], [])


def test_parse_relations_non_list_reports_problem():
    relations, problems = parse_relations({"relations": "works_at acme"})
    assert relations == []
    assert len(problems) == 1 and "expected a list" in problems[0]


def test_parse_relations_non_dict_entries_skipped_with_problem():
    relations, problems = parse_relations({"relations": ["works_at", 42]})
    assert relations == []
    assert len(problems) == 2
    assert all("expected a mapping" in p for p in problems)


def test_parse_relations_missing_rel_or_target_skipped():
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


def test_parse_relations_unknown_rel_excluded_but_reported():
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


def test_parse_relations_yaml_dates_pass_through_as_iso_strings():
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


def test_upsert_adds_new_relation_and_preserves_body():
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


def test_upsert_identical_open_relation_is_noop():
    relation = Relation(rel="works_at", target="organisations/acme", valid_from="2025-03-01")
    once, _ = upsert_relation_in_text(PERSON_NOTE, relation)
    twice, action = upsert_relation_in_text(once, relation)
    assert action == "noop"
    assert twice == once                            # text unchanged, byte-for-byte


def test_upsert_noop_matches_tolerant_target_forms():
    once, _ = upsert_relation_in_text(
        PERSON_NOTE, Relation(rel="works_at", target="organisations/acme")
    )
    _, action = upsert_relation_in_text(
        once, Relation(rel="works_at", target="[[knowledge/organisations/acme]]")
    )
    assert action == "noop"


def test_upsert_closes_open_relation_keeping_valid_from():
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


def test_upsert_after_close_appends_new_entry_history_intact():
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
    assert len(fm["relations"]) == 2               # closed span + new open one
    assert fm["relations"][0]["valid_until"] == "2026-02-28"
    assert "valid_until" not in fm["relations"][1]


def test_upsert_repeated_close_without_valid_from_is_noop():
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
    assert len(fm["relations"]) == 1                # no duplicate appended
    assert fm["relations"][0]["valid_from"] == "2022-01-01"
    assert fm["relations"][0]["valid_until"] == "2025-02-28"


def test_upsert_close_with_explicit_different_valid_from_still_distinguishes():
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
    assert len(fm["relations"]) == 2


def test_upsert_on_note_without_frontmatter_creates_fence():
    plain = "# Anna\n\nJust a body.\n"
    new_text, action = upsert_relation_in_text(
        plain, Relation(rel="met_at", target="meetings/2026/2026-06-12-kern-call")
    )
    assert action == "added"
    assert new_text.startswith("---\n")
    assert new_text.endswith(plain)                 # body preserved byte-for-byte
    fm, body = _split_frontmatter(new_text)
    assert body == plain
    assert fm["relations"][0]["rel"] == "met_at"


# ---------------------------------------------------------------------------
# append_fact_to_log
# ---------------------------------------------------------------------------

def test_append_fact_to_existing_log_section_lands_last():
    result = append_fact_to_log(PERSON_NOTE, "2026-06-12 — joined ACME")
    assert "- 2026-06-01 — met at the Kern kickoff\n- 2026-06-12 — joined ACME" in result


def test_append_fact_inserts_before_following_section():
    note = "# T\n\n## Log\n\n- old\n\n## Open questions\n\n- q1\n"
    result = append_fact_to_log(note, "new fact")
    assert "- old\n- new fact" in result
    assert result.index("- new fact") < result.index("## Open questions")
    assert result.endswith("## Open questions\n\n- q1\n")


def test_append_fact_creates_log_section_at_end_when_absent():
    note = "---\ntitle: x\n---\n\n# X\n\nBody.\n"
    result = append_fact_to_log(note, "first fact")
    assert result.endswith("Body.\n\n## Log\n\n- first fact\n")


def test_append_fact_to_empty_log_section():
    note = "# T\n\n## Log\n"
    result = append_fact_to_log(note, "only fact")
    assert "## Log\n\n- only fact" in result


def test_append_fact_idempotent_on_exact_duplicate():
    # Crash-/retry-safety: re-applying the EXACT same fact line is a no-op,
    # so a consolidation rerun or a retried append never double-applies it.
    once = append_fact_to_log(PERSON_NOTE, "2026-06-12 — joined ACME")
    twice = append_fact_to_log(once, "2026-06-12 — joined ACME")
    assert twice == once                            # text unchanged
    assert once.count("- 2026-06-12 — joined ACME") == 1


def test_append_fact_idempotent_across_crlf_line_endings():
    # F137: a note that picked up CRLF endings (Windows editor/sync tool)
    # must still dedup, or the crash/retry-safety silently fails.
    crlf = "# T\r\n\r\n## Log\r\n\r\n- 2026-06-12 — joined ACME\r\n"
    result = append_fact_to_log(crlf, "2026-06-12 — joined ACME")
    assert result == crlf                           # no second bullet added


def test_append_fact_distinct_facts_both_append():
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


def _history_vault(tmp_path: Path):
    root = _vault(tmp_path)
    _write(root, "knowledge/people/anna.md", _ANNA_HISTORY)
    _write(root, "knowledge/organisations/acme.md", "---\ntitle: Acme\n---\n")
    _write(root, "knowledge/organisations/initech.md", "---\ntitle: Initech\n---\n")
    return paths_for_root(root)


def test_query_relations_open_only_by_default(tmp_path: Path):
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    hits = query_relations(paths)
    assert [(h.rel, h.target) for h in hits] == [("works_at", "organisations/initech")]


def test_query_relations_include_closed(tmp_path: Path):
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    hits = query_relations(paths, include_closed=True)
    targets = {h.target for h in hits}
    assert targets == {"organisations/acme", "organisations/initech"}


def test_query_relations_as_of_selects_historical_interval(tmp_path: Path):
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    # Mid-2025 -> Anna was at Acme, not Initech.
    hits = query_relations(paths, as_of="2025-06-01")
    assert [(h.target) for h in hits] == ["organisations/acme"]


def test_query_relations_reverse_lookup_by_target(tmp_path: Path):
    from ingest_lib.relations import query_relations
    paths = _history_vault(tmp_path)
    # Who works at initech (open)? tolerant of the knowledge/ + .md forms.
    hits = query_relations(paths, target="knowledge/organisations/initech.md")
    assert [h.entity for h in hits] == ["people/anna"]
    assert hits[0].source == "knowledge/meetings/2026/2026-01-05-x"
