"""Unit tests for the deterministic concept-relationship logic.

These cover the pure functions only — co-occurrence counting, semantic
thresholding, and related-map ranking. The numpy/embeddings glue and the
file I/O are exercised by running the real pipeline, not here.
"""
from __future__ import annotations

import math

from ingest_lib.connections import (
    Edge,
    Related,
    build_related_map,
    cooccurrence_edges,
    semantic_edges,
)
from ingest_lib.metadata import IndexRecord


def _rec(path: str, topics: list[str], created_at: str = "") -> IndexRecord:
    return IndexRecord(
        relative_path=path,
        source_hash="h-" + path,
        size_bytes=1,
        extension=".md",
        extractor="text",
        status="processed",
        raw_path="archive/raw/" + path,
        processed_path="archive/processed/" + path,
        index_note_path=None,
        topics=topics,
        created_at=created_at,
    )


def _normalize(vec: tuple[float, ...]) -> tuple[float, ...]:
    mag = math.sqrt(sum(x * x for x in vec))
    return tuple(x / mag for x in vec)


# ---------------------------------------------------------------------------
# co-occurrence
# ---------------------------------------------------------------------------

def test_cooccurrence_counts_shared_topics_across_documents():
    recs = [
        _rec("a.md", ["Alpha", "Beta"]),
        _rec("b.md", ["Beta", "Gamma"]),
        _rec("c.md", ["Alpha", "Beta", "Gamma"]),
    ]
    by_pair = {(e.a, e.b): e for e in cooccurrence_edges(recs)}

    assert by_pair[("alpha", "beta")].weight == 2.0
    assert set(by_pair[("alpha", "beta")].sources) == {"a.md", "c.md"}
    assert by_pair[("beta", "gamma")].weight == 2.0
    assert by_pair[("alpha", "gamma")].weight == 1.0
    assert by_pair[("alpha", "gamma")].sources == ("c.md",)
    assert all(e.kind == "cooccurrence" for e in by_pair.values())


def test_cooccurrence_edges_are_sorted_and_deterministic():
    edges = cooccurrence_edges([_rec("a.md", ["Zeta", "Alpha", "Mu"])])
    pairs = [(e.a, e.b) for e in edges]
    assert pairs == [("alpha", "mu"), ("alpha", "zeta"), ("mu", "zeta")]


def test_single_or_zero_topic_documents_yield_no_edges():
    assert cooccurrence_edges([_rec("a.md", ["Solo"])]) == []
    assert cooccurrence_edges([_rec("a.md", [])]) == []


def test_topic_case_and_punctuation_collapse_to_one_concept():
    recs = [
        _rec("a.md", ["Packet Switching", "TCP"]),
        _rec("b.md", ["packet-switching", "TCP"]),
    ]
    by_pair = {(e.a, e.b): e for e in cooccurrence_edges(recs)}
    assert by_pair[("packet-switching", "tcp")].weight == 2.0


def test_duplicate_topic_within_one_document_does_not_inflate_weight():
    # Same concept twice in one doc (case drift) must not create a self-pair
    # or double-count.
    recs = [_rec("a.md", ["Graphs", "graphs", "Trees"])]
    by_pair = {(e.a, e.b): e for e in cooccurrence_edges(recs)}
    assert ("graphs", "graphs") not in by_pair
    assert by_pair[("graphs", "trees")].weight == 1.0


# ---------------------------------------------------------------------------
# semantic (cosine on normalized concept vectors)
# ---------------------------------------------------------------------------

def test_semantic_edges_link_nearest_neighbours_above_floor():
    vecs = {
        "a": _normalize((1.0, 0.0)),
        "b": _normalize((0.99, 0.05)),   # very close to a
        "c": _normalize((0.0, 1.0)),     # far from both
    }
    edges = semantic_edges(vecs, top_k=2, min_cosine=0.5)
    pairs = {(e.a, e.b) for e in edges}
    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs
    assert ("b", "c") not in pairs       # c's best (0.05) is below the floor
    assert all(e.kind == "semantic" for e in edges)


def test_semantic_edges_keep_only_top_k_per_concept():
    # Collinear-ish points on a line: a closest to b, b to a/c, c to b/d, d to c.
    vecs = {
        "a": _normalize((1.0, 0.0)),
        "b": _normalize((1.0, 0.10)),
        "c": _normalize((1.0, 0.30)),
        "d": _normalize((1.0, 0.60)),
    }
    pairs = {(e.a, e.b) for e in semantic_edges(vecs, top_k=1, min_cosine=0.0)}
    # Each concept keeps only its single nearest neighbour (symmetrised).
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert ("c", "d") in pairs
    assert ("a", "d") not in pairs       # never mutual-nearest


def test_semantic_edges_carry_cosine_weight_and_are_sorted():
    vecs = {"a": _normalize((1.0, 0.0)), "b": _normalize((1.0, 0.0))}
    edges = semantic_edges(vecs, top_k=5, min_cosine=0.0)
    assert [(e.a, e.b) for e in edges] == [("a", "b")]
    assert edges[0].kind == "semantic"
    assert edges[0].weight == 1.0
    assert edges[0].sources == ()


# ---------------------------------------------------------------------------
# related-map ranking
# ---------------------------------------------------------------------------

def test_related_map_merges_signals_and_ranks_multi_signal_first():
    edges = [
        Edge("a", "b", "cooccurrence", 3.0, ("x.md",)),
        Edge("a", "b", "semantic", 0.9, ()),
        Edge("a", "c", "cooccurrence", 5.0, ("y.md",)),
    ]
    displays = {"a": "Alpha", "b": "Beta", "c": "Cee"}
    rel = build_related_map(edges, displays, top_n=10)

    a = rel["a"]
    # b has BOTH signals -> ranks above c even though c's co-occurrence is higher
    assert a[0].slug == "b"
    assert set(a[0].kinds) == {"cooccurrence", "semantic"}
    assert a[0].display == "Beta"
    assert a[1].slug == "c"
    # edges are symmetric
    assert rel["b"][0].slug == "a"
    assert rel["c"][0].slug == "a"


def test_related_map_caps_neighbors_to_top_n():
    edges = [
        Edge("a", "b", "cooccurrence", 5.0, ()),
        Edge("a", "c", "cooccurrence", 3.0, ()),
        Edge("a", "d", "cooccurrence", 1.0, ()),
    ]
    rel = build_related_map(edges, {}, top_n=2)
    assert [r.slug for r in rel["a"]] == ["b", "c"]


def test_related_falls_back_to_slug_when_display_missing():
    edges = [Edge("a", "b", "cooccurrence", 1.0, ())]
    rel = build_related_map(edges, {}, top_n=5)
    assert rel["a"][0].display == "b"
    assert isinstance(rel["a"][0], Related)
