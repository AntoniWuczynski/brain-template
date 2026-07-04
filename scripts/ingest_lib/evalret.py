"""Retrieval-quality scoring: recall@k and MRR over a golden query set.

Pure and deterministic. A ``retrieve`` callable maps ``query -> ordered
list of source paths`` (best first, de-duplicated per source); the scorer
compares that against each golden query's ``expected`` source paths. Keeping
this separate from the CLI (``scripts/eval_retrieval.py``) makes it testable
without a model load and lets any retrieval backend be evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class QueryResult:
    query: str
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]        # ordered source paths (de-duped)
    recall: dict[int, float] = field(default_factory=dict)  # k -> recall@k
    first_hit_rank: int | None = None  # 1-based rank of the first expected hit
    reciprocal_rank: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class EvalReport:
    results: tuple[QueryResult, ...]
    ks: tuple[int, ...]

    def recall_at(self, k: int) -> float:
        """Mean recall@k across queries (0.0 when there are no queries)."""
        if not self.results:
            return 0.0
        return sum(r.recall.get(k, 0.0) for r in self.results) / len(self.results)

    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    def misses(self) -> list[QueryResult]:
        """Queries that found NONE of their expected sources (in any rank)."""
        return [r for r in self.results if r.first_hit_rank is None]


def _score_one(
    query: str, expected: Sequence[str], retrieved: Sequence[str],
    *, ks: Sequence[int], note: str = "",
) -> QueryResult:
    expected_set = set(expected)
    # 1-based rank of the first retrieved path that is expected.
    first_rank: int | None = None
    for i, path in enumerate(retrieved, start=1):
        if path in expected_set:
            first_rank = i
            break
    recall: dict[int, float] = {}
    denom = len(expected_set) or 1
    for k in ks:
        found = len(expected_set & set(retrieved[:k]))
        recall[k] = found / denom
    return QueryResult(
        query=query,
        expected=tuple(expected),
        retrieved=tuple(retrieved),
        recall=recall,
        first_hit_rank=first_rank,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        note=note,
    )


def evaluate(
    golden: Sequence[dict],
    retrieve: Callable[[str, int], list[str]],
    *,
    ks: Sequence[int] = (5, 10),
    fetch: int | None = None,
) -> EvalReport:
    """Run every golden query through ``retrieve`` and score it.

    ``golden`` items are ``{"query": str, "expected": [paths], "note"?: str}``.
    ``retrieve(query, n)`` returns up to ``n`` ordered source paths.
    ``fetch`` is how many results to request (defaults to ``max(ks)``).
    """
    n = fetch if fetch is not None else (max(ks) if ks else 10)
    results = [
        _score_one(
            g["query"], g.get("expected", []), retrieve(g["query"], n),
            ks=ks, note=g.get("note", ""),
        )
        for g in golden
    ]
    return EvalReport(results=tuple(results), ks=tuple(ks))
