"""Retrieval eval scoring (P2) — pure, no model load."""
from __future__ import annotations

from ingest_lib.evalret import evaluate


def _fixed(mapping):
    return lambda q, n: mapping.get(q, [])[:n]


def test_recall_and_mrr_on_hits():
    golden = [
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


def test_partial_recall_and_miss():
    golden = [
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


def test_recall_at_k_respects_cutoff():
    golden = [{"query": "a", "expected": ["target"]}]
    # target sits at rank 7 -> in recall@10 but not recall@5.
    retrieve = _fixed({"a": [f"d{i}" for i in range(6)] + ["target"]})
    rep = evaluate(golden, retrieve, ks=(5, 10), fetch=10)
    assert rep.recall_at(5) == 0.0
    assert rep.recall_at(10) == 1.0


def test_empty_golden_is_zero_not_crash():
    rep = evaluate([], _fixed({}), ks=(5,))
    assert rep.recall_at(5) == 0.0 and rep.mrr() == 0.0


def test_retriever_overfetches_chunks_and_returns_n_distinct_sources(monkeypatch):
    # The retriever de-dupes CHUNKS to SOURCES: a source with many chunks must
    # not crowd distinct sources out of the top-n it reports to the scorer.
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval
    from ingest_lib.semantic import SearchHit

    def _hit(src: str, idx: int) -> SearchHit:
        return SearchHit(score=1.0, source_relative_path=src, title=src,
                         chunk_idx=idx, snippet="s", origin="")

    # First 5 chunks all belong to one source; the 6th is a second source.
    chunks = [_hit("big.pdf", i) for i in range(5)] + [_hit("other.pdf", 0)]
    captured: dict[str, int] = {}

    def fake_search(paths, query, *, top_k, **kw):
        captured["top_k"] = top_k
        return chunks

    monkeypatch.setattr(eval_retrieval, "semantic_search", fake_search)
    retrieve = eval_retrieval._retriever(paths=None, top_k=10)
    got = retrieve("q", 2)

    assert got == ["big.pdf", "other.pdf"]   # 2 distinct sources, not 1
    assert captured["top_k"] > 2             # over-fetched chunks, not just n
