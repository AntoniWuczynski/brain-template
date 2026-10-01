"""Section-level retrieval scoring: does a strategy land in the right SECTION
of a known long document, not just the right document?

Pure and deterministic, like ``evalret``. Each strategy yields a ranked list
of candidate chunk sets — one chunk per search hit, a whole chunk range per
outline section — and a candidate is a hit when it overlaps the golden
query's expected chunks. The CLI is ``scripts/eval_sections.py``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

from .outline import Section


class SectionGolden(TypedDict):
    """One golden line: the query, the source it is about, and a substring
    of the heading path of the section(s) that answer it."""
    query: str
    source: str
    expected_heading: str
    note: NotRequired[str]


def expected_chunks(sections: Sequence[Section], needle: str) -> frozenset[int]:
    """Every chunk of every section whose heading path contains ``needle``
    (case-insensitive) — a heading can recur, and each occurrence counts."""
    n = needle.casefold()
    return frozenset(
        i
        for s in sections
        if n in s.heading_path.casefold()
        for i in range(s.first_chunk, s.last_chunk + 1)
    )


def section_chunks(section: Section) -> frozenset[int]:
    return frozenset(range(section.first_chunk, section.last_chunk + 1))


def first_hit_rank(ranked: Sequence[frozenset[int]], expected: frozenset[int]) -> int | None:
    """1-based rank of the first candidate overlapping ``expected``."""
    for rank, chunks in enumerate(ranked, start=1):
        if chunks & expected:
            return rank
    return None


def render_outline(sections: Sequence[Section]) -> str:
    """Numbered outline lines, the form an LLM navigator is shown."""
    return "\n".join(
        f"{n}. {s.heading_path or '(no heading)'} | {s.preview}"
        for n, s in enumerate(sections, start=1)
    )


@dataclass(frozen=True)
class SectionResult:
    query: str
    source: str
    expected_heading: str
    ranks: dict[str, int | None] = field(default_factory=dict)  # strategy -> first-hit rank


def hit_rate(results: Sequence[SectionResult], strategy: str, k: int) -> float:
    """Share of queries whose first hit for ``strategy`` is within rank ``k``
    (0.0 when no query was scored for it)."""
    scored = [r.ranks[strategy] for r in results if strategy in r.ranks]
    if not scored:
        return 0.0
    return sum(1 for rank in scored if rank is not None and rank <= k) / len(scored)
