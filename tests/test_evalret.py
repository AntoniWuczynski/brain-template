"""Retrieval eval scoring (P2) — pure, no model load."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ingest_lib.config import paths_for_root
from ingest_lib.evalret import GoldenQuery, evaluate


def _fixed(mapping: dict[str, list[str]]) -> Callable[[str, int], list[str]]:
    return lambda q, n: mapping.get(q, [])[:n]


def test_recall_and_mrr_on_hits() -> None:
    golden: list[GoldenQuery] = [
        {"query": "a", "expected": ["x", "y"]},
        {"query": "b", "expected": ["z"]},
    ]
    retrieve = _fixed({
        "a": ["x", "w", "y", "v"],   # both expected in top-3
        "b": ["p", "z"],             # z at rank 2
    })
    rep = evaluate(golden, retrieve, ks=(5, 10), fetch=10)
    # a: recall@5 = 2/2 = 1.0, first hit rank 1 -> RR 1.0
    # b: recall@5 = 1/1 = 1.0, first hit rank 2 -> RR 0.5
    assert rep.recall_at(5) == 1.0
    assert rep.mrr() == (1.0 + 0.5) / 2
    assert not rep.misses()


def test_partial_recall_and_miss() -> None:
    golden: list[GoldenQuery] = [
        {"query": "a", "expected": ["x", "y"]},   # only x retrieved
        {"query": "b", "expected": ["z"]},        # never retrieved
    ]
    retrieve = _fixed({"a": ["x", "w"], "b": ["p", "q"]})
    rep = evaluate(golden, retrieve, ks=(5,), fetch=5)
    a = next(r for r in rep.results if r.query == "a")
    assert a.recall[5] == 0.5           # 1 of 2 expected
    assert a.first_hit_rank == 1
    b = next(r for r in rep.results if r.query == "b")
    assert b.first_hit_rank is None
    assert b.reciprocal_rank == 0.0
    assert [r.query for r in rep.misses()] == ["b"]


def test_recall_at_k_respects_cutoff() -> None:
    golden: list[GoldenQuery] = [{"query": "a", "expected": ["target"]}]
    # target sits at rank 7 -> in recall@10 but not recall@5.
    retrieve = _fixed({"a": [f"d{i}" for i in range(6)] + ["target"]})
    rep = evaluate(golden, retrieve, ks=(5, 10), fetch=10)
    assert rep.recall_at(5) == 0.0
    assert rep.recall_at(10) == 1.0


def test_empty_golden_is_zero_not_crash() -> None:
    rep = evaluate([], _fixed({}), ks=(5,))
    assert rep.recall_at(5) == 0.0 and rep.mrr() == 0.0


def test_retriever_overfetches_chunks_and_returns_n_distinct_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The retriever de-dupes CHUNKS to SOURCES: a source with many chunks must
    # not crowd distinct sources out of the top-n it reports to the scorer.
    import logging
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval
    from ingest_lib.config import VaultPaths
    from ingest_lib.semantic import SearchHit

    def _hit(src: str, idx: int) -> SearchHit:
        return SearchHit(score=1.0, source_relative_path=src, title=src,
                         chunk_idx=idx, snippet="s", origin="")

    # First 5 chunks all belong to one source; the 6th is a second source.
    chunks = [_hit("big.pdf", i) for i in range(5)] + [_hit("other.pdf", 0)]
    captured: dict[str, int] = {}

    def fake_search(
        paths: VaultPaths, query: str, *, top_k: int, mode: str = "hybrid",
        logger: logging.Logger | None = None,
    ) -> list[SearchHit]:
        captured["top_k"] = top_k
        return chunks

    monkeypatch.setattr(eval_retrieval, "semantic_search", fake_search)
    # semantic_search is stubbed above and never dereferences paths.
    retrieve = eval_retrieval._retriever(paths=paths_for_root(tmp_path), top_k=10)
    got = retrieve("q", 2)

    assert got == ["big.pdf", "other.pdf"]   # 2 distinct sources, not 1
    assert captured["top_k"] > 2             # over-fetched chunks, not just n


def test_retriever_escalates_fetch_when_pool_lacks_n_distinct_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # A skewed pool where the first fetch (5x) surfaces only 1 distinct
    # source must escalate until n distinct sources are found, rather than
    # silently returning fewer than n (scoring recall@n as if the pool were
    # exhaustive when it wasn't).
    import logging
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval
    from ingest_lib.config import VaultPaths
    from ingest_lib.semantic import SearchHit

    def _hit(src: str, idx: int) -> SearchHit:
        return SearchHit(score=1.0, source_relative_path=src, title=src,
                         chunk_idx=idx, snippet="s", origin="")

    # A fixed, oversized ranking: "dominant.pdf" crowds every rank except one
    # early "second.pdf" and one "third.pdf" that only enters the pool once
    # it's fetched deep enough (rank 81) — so 3 distinct sources require
    # escalating well past the initial 5x fetch (50).
    universe = (
        [_hit("dominant.pdf", i) for i in range(49)]
        + [_hit("second.pdf", 0)]
        + [_hit("dominant.pdf", 49 + i) for i in range(30)]
        + [_hit("third.pdf", 0)]
        + [_hit("dominant.pdf", 80 + i) for i in range(900)]
    )
    calls: list[int] = []

    def fake_search(
        paths: VaultPaths, query: str, *, top_k: int, mode: str = "hybrid",
        logger: logging.Logger | None = None,
    ) -> list[SearchHit]:
        calls.append(top_k)
        return universe[:top_k]

    monkeypatch.setattr(eval_retrieval, "semantic_search", fake_search)
    retrieve = eval_retrieval._retriever(paths=paths_for_root(tmp_path), top_k=10)
    got = retrieve("q", 3)

    assert got == ["dominant.pdf", "second.pdf", "third.pdf"]
    assert len(calls) > 1                 # it escalated past the initial 5x fetch
    assert calls == sorted(calls)         # each retry asked for more, not less


def test_load_golden_rejects_empty_expected(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text('{"query": "q", "expected": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        eval_retrieval._load_golden(golden_path)


def test_load_golden_rejects_missing_query(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text('{"expected": ["x.md"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="query"):
        eval_retrieval._load_golden(golden_path)


def test_load_golden_accepts_valid_lines(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        '{"query": "q", "expected": ["x.md"], "note": "n"}\n'
        '# a comment line\n'
        '\n',
        encoding="utf-8",
    )
    loaded = eval_retrieval._load_golden(golden_path)
    assert loaded == [{"query": "q", "expected": ["x.md"], "note": "n"}]
