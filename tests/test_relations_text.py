"""Tests for ingest_lib.relations — the pure text-editing helpers the MCP
entity tools build on: ``upsert_relation_in_text`` (splicing a relation into
a note's frontmatter without disturbing anything else) and
``append_fact_to_log`` (appending a dated bullet to a note's Log section)."""
from __future__ import annotations

import pytest

from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import Relation, append_fact_to_log, upsert_relation_in_text


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
