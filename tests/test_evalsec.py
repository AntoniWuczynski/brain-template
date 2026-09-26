"""ingest_lib.evalsec: section-level scoring against expected chunk sets."""
from __future__ import annotations

from ingest_lib.evalsec import (
    SectionResult,
    expected_chunks,
    first_hit_rank,
    hit_rate,
    render_outline,
)
from ingest_lib.outline import Section


def _sec(heading: str, first: int, last: int) -> Section:
    return Section(heading_path=heading, first_chunk=first, last_chunk=last, chars=1, preview="p")


def test_expected_chunks_matches_every_occurrence_case_insensitively() -> None:
    secs = [_sec("Demo", 0, 1), _sec("Key idea", 2, 2), _sec("A > demo", 3, 4)]
    assert expected_chunks(secs, "DEMO") == frozenset({0, 1, 3, 4})
    assert expected_chunks(secs, "missing") == frozenset()


def test_first_hit_rank_is_one_based_overlap() -> None:
    ranked = [frozenset({9}), frozenset(), frozenset({2, 3})]
    assert first_hit_rank(ranked, frozenset({3})) == 3
    assert first_hit_rank(ranked, frozenset({7})) is None


def test_hit_rate_ignores_queries_without_that_strategy() -> None:
    results = [
        SectionResult("a", "s", "h", {"search": 1, "outline-llm": 4}),
        SectionResult("b", "s", "h", {"search": None}),
    ]
    assert hit_rate(results, "search", 1) == 0.5
    assert hit_rate(results, "outline-llm", 3) == 0.0
    assert hit_rate(results, "outline-llm", 5) == 1.0
    assert hit_rate(results, "absent", 5) == 0.0


def test_render_outline_numbers_from_one() -> None:
    assert render_outline([_sec("", 0, 0), _sec("B", 1, 1)]) == "1. (no heading) | p\n2. B | p"
