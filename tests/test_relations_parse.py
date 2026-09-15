"""Tests for ingest_lib.relations — frontmatter parsing tolerance
(``parse_relations``), building the entity graph from notes on disk
(``entity_notes``), and the as-of/reverse-lookup query layer
(``query_relations``, P3)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.relations import RELATION_VOCAB, Relation, entity_notes, parse_relations


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
