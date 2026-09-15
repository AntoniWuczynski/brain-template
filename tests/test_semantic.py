"""Tests for the optional ranking layers as they land in ``semantic.search``.

The real embedding model is never loaded: the index is seeded by hand with a
deterministic hash encoder (the pattern ``test_semantic_upsert.py`` uses) and
the cross-encoder is a stub. Every test asserts against the FUSED order, so a
layer that silently did nothing would fail here rather than pass quietly.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ingest_lib import rank, semantic, summarize
from ingest_lib.config import VaultPaths, paths_for_root

_LOG = logging.getLogger("test")
_DIM = 8


def _fake_encode(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:_DIM], dtype=np.uint8).astype(np.float32) + 1.0
        rows.append(vec / np.linalg.norm(vec))
    return np.vstack(rows)


class _FakeEmbedder:
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        return _fake_encode(sentences)


# (source_relative_path, chunk text, origin)
_ROWS: list[tuple[str, str, str]] = [
    ("uni/arch/lecture8.pdf", "cpu caches and the memory hierarchy", "pdf-mineru"),
    ("uni/arch/lecture9.pdf", "cpu caches write-back policy and pipelining", "pdf-mineru"),
    ("uni/poetry/rhyme.pdf", "rhyme scheme and metre in poetry", "pdf-mineru"),
    ("knowledge/people/anna.md", "anna works on the caches project", "knowledge-note"),
    ("knowledge/notes/misc.md", "an unrelated curated note about gardening", "knowledge-note"),
]


def _vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    semantic._write_index_files(
        _fake_encode([text for _rel, text, _o in _ROWS]),
        [
            semantic.MetaRow(
                source_relative_path=rel,
                source_hash="h-" + rel,
                title=Path(rel).stem,
                chunk_idx=0,
                text=text,
                origin=origin,
                heading_path="",
                model="BAAI/bge-small-en-v1.5",
                gen="",
            )
            for rel, text, origin in _ROWS
        ],
        vectors_path=paths.metadata / "embeddings.npy",
        meta_path=paths.metadata / "embeddings_meta.jsonl",
    )
    monkeypatch.setenv("BRAIN_QUERY_INSTRUCTION", "0")
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeEmbedder(), "cpu"))
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
    monkeypatch.setattr(semantic, "_LEXICAL_CACHE", None)
    monkeypatch.setattr(rank, "_GRAPH_CACHE", None)
    return paths


def _connections(paths: VaultPaths, edges: list[dict[str, object]]) -> None:
    (paths.metadata / "connections.jsonl").write_text(
        "\n".join(json.dumps(e) for e in edges) + "\n", encoding="utf-8"
    )


def _order(paths: VaultPaths, query: str, *, mode: str = "hybrid") -> list[str]:
    return [
        h.source_relative_path
        for h in semantic.search(paths, query, top_k=5, mode=mode, logger=_LOG)
    ]


_QUERY = "cpu caches"


def _fake_paraphrases(**kwargs: object) -> object:
    """The expansion LLM, stubbed: the layer's determinism has to hold over
    the real ``rank.expand_query`` (cache file included), not over a stub of
    it."""
    schema = kwargs["schema"]
    assert isinstance(schema, type)
    return schema(paraphrases=["processor cache memory", "cpu memory hierarchy"])


# ---------------------------------------------------------------------------
# off by default
# ---------------------------------------------------------------------------

def test_every_ranking_layer_is_off_unless_the_environment_switches_it_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live server must rank exactly as it did until the owner flips a
    flag: an unset environment and an explicitly-disabled one agree."""
    paths = _vault(tmp_path, monkeypatch)
    _connections(paths, [
        {"a": "caches", "b": "pipelining", "kind": "cooccurrence", "weight": 3.0,
         "sources": ["uni/arch/lecture8.pdf", "uni/arch/lecture9.pdf"]},
    ])
    for var in ("BRAIN_RANK_GRAPH", "BRAIN_RANK_SALIENCE", "BRAIN_RANK_RERANK",
                "BRAIN_QUERY_EXPANSION"):
        monkeypatch.delenv(var, raising=False)
    default = _order(paths, _QUERY)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a ranking layer ran while every flag was off")

    monkeypatch.setattr(rank, "load_graph", _boom)
    monkeypatch.setattr(rank, "load_reranker", _boom)
    monkeypatch.setattr(rank, "expand_query", _boom)
    assert _order(paths, _QUERY) == default
    for var in ("BRAIN_RANK_GRAPH", "BRAIN_RANK_SALIENCE", "BRAIN_RANK_RERANK",
                "BRAIN_QUERY_EXPANSION"):
        monkeypatch.setenv(var, "0")
    assert _order(paths, _QUERY) == default


def test_the_layers_are_hybrid_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dense and lexical return raw cosine / BM25 scores, whose meaning an
    additive RRF-scale boost would silently change."""
    paths = _vault(tmp_path, monkeypatch)
    _connections(paths, [
        {"a": "caches", "b": "pipelining", "kind": "cooccurrence", "weight": 3.0,
         "sources": ["uni/arch/lecture8.pdf", "uni/poetry/rhyme.pdf"]},
    ])
    before = {m: _order(paths, _QUERY, mode=m) for m in ("dense", "lexical")}
    monkeypatch.setenv("BRAIN_RANK_GRAPH", "1")
    monkeypatch.setenv("BRAIN_RANK_SALIENCE", "1")
    monkeypatch.setattr(
        rank, "load_reranker",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerank ran in a non-hybrid mode")),
    )
    assert {m: _order(paths, _QUERY, mode=m) for m in ("dense", "lexical")} == before


# ---------------------------------------------------------------------------
# graph adjacency
# ---------------------------------------------------------------------------

def test_graph_boost_promotes_a_source_adjacent_to_the_top_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    _connections(paths, [
        {"a": "caches", "b": "memory", "kind": "cooccurrence", "weight": 3.0,
         "sources": ["uni/arch/lecture8.pdf", "uni/poetry/rhyme.pdf"]},
    ])
    baseline = _order(paths, _QUERY)
    monkeypatch.setenv("BRAIN_RANK_GRAPH", "1")
    monkeypatch.setenv("BRAIN_RANK_GRAPH_WEIGHT", "50")   # mechanism, not tuning
    boosted = _order(paths, _QUERY)
    assert boosted != baseline
    # rhyme.pdf shares a co-occurrence source-set with the top hit and nothing
    # else does, so it is the one the graph lifts.
    assert boosted.index("uni/poetry/rhyme.pdf") < baseline.index("uni/poetry/rhyme.pdf")


def test_graph_boost_is_inert_without_a_connections_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    baseline = _order(paths, _QUERY)
    monkeypatch.setenv("BRAIN_RANK_GRAPH", "1")
    assert _order(paths, _QUERY) == baseline


# ---------------------------------------------------------------------------
# salience
# ---------------------------------------------------------------------------

def test_salience_lifts_a_curated_note_without_dropping_archive_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    baseline = semantic.search(paths, _QUERY, top_k=5, logger=_LOG)
    monkeypatch.setenv("BRAIN_RANK_SALIENCE", "1")
    monkeypatch.setenv("BRAIN_RANK_SALIENCE_WEIGHT", "50")
    after = semantic.search(paths, _QUERY, top_k=5, logger=_LOG)
    by_path = {h.source_relative_path: h.score for h in after}
    base_by_path = {h.source_relative_path: h.score for h in baseline}
    # Curated notes gain; archive chunks keep exactly the score they had.
    assert by_path["knowledge/people/anna.md"] > base_by_path["knowledge/people/anna.md"]
    assert by_path["uni/arch/lecture8.pdf"] == pytest.approx(
        base_by_path["uni/arch/lecture8.pdf"]
    )
    # Every archive source that was retrieved is still retrieved.
    assert {p for p in base_by_path if not p.startswith("knowledge/")} <= set(by_path)


# ---------------------------------------------------------------------------
# cross-encoder rerank
# ---------------------------------------------------------------------------

class _ReverseEncoder:
    """Scores the window in reverse order, so any real rerank is visible.

    Structural, not a subclass: ``rank.CrossEncoderModel`` is a Protocol, so
    the real ``CrossEncoder`` (which subclasses nothing here) and this double
    satisfy it the same way."""

    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]:
        return [float(i) for i in range(len(sentences))]


def test_rerank_permutes_the_window_and_keeps_the_rrf_score_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window's own fused scores are re-assigned in the new order, so
    ``SearchHit.score`` stays the documented rank score and
    ``recency.memory_search``'s multiplication keeps its meaning."""
    paths = _vault(tmp_path, monkeypatch)
    baseline = semantic.search(paths, _QUERY, top_k=5, logger=_LOG)
    monkeypatch.setenv("BRAIN_RANK_RERANK", "1")
    monkeypatch.setattr(rank, "load_reranker", lambda *a, **k: _ReverseEncoder())
    after = semantic.search(paths, _QUERY, top_k=5, logger=_LOG)
    assert [h.source_relative_path for h in after] == [
        h.source_relative_path for h in reversed(baseline)
    ]
    assert [h.score for h in after] == [h.score for h in baseline]


def test_rerank_falls_back_to_the_fused_order_when_the_model_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    baseline = _order(paths, _QUERY)
    monkeypatch.setenv("BRAIN_RANK_RERANK", "1")
    monkeypatch.setattr(rank, "load_reranker", lambda *a, **k: None)
    assert _order(paths, _QUERY) == baseline


def test_rerank_only_touches_the_configured_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    baseline = _order(paths, _QUERY)
    monkeypatch.setenv("BRAIN_RANK_RERANK", "1")
    monkeypatch.setenv("BRAIN_RANK_RERANK_TOP_N", "2")
    monkeypatch.setattr(rank, "load_reranker", lambda *a, **k: _ReverseEncoder())
    after = _order(paths, _QUERY)
    assert after[:2] == list(reversed(baseline[:2]))
    assert after[2:] == baseline[2:]


# ---------------------------------------------------------------------------
# query expansion
# ---------------------------------------------------------------------------

def test_query_expansion_fuses_a_paraphrase_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    baseline = _order(paths, _QUERY)
    assert baseline.index("uni/poetry/rhyme.pdf") > 0
    monkeypatch.setenv("BRAIN_QUERY_EXPANSION", "1")
    monkeypatch.setenv("BRAIN_QUERY_EXPANSION_WEIGHT", "1")
    monkeypatch.setattr(rank, "expand_query", lambda *a, **k: ["rhyme scheme and metre in poetry"])
    after = _order(paths, _QUERY)
    assert after.index("uni/poetry/rhyme.pdf") < baseline.index("uni/poetry/rhyme.pdf")


def test_query_expansion_that_returns_nothing_leaves_the_ranking_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, monkeypatch)
    baseline = _order(paths, _QUERY)
    monkeypatch.setenv("BRAIN_QUERY_EXPANSION", "1")
    monkeypatch.setattr(rank, "expand_query", lambda *a, **k: [])
    assert _order(paths, _QUERY) == baseline


def test_query_expansion_does_not_pay_for_a_degraded_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hybrid search whose embedder fails to load degrades to BM25-only and
    returns without ever fusing the paraphrases. Asking for them first paid a
    real LLM call for work that was thrown away — and logged it as a success."""
    paths = _vault(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAIN_QUERY_EXPANSION", "1")

    def _no_weights() -> tuple[semantic.Embedder, str]:
        raise RuntimeError("no sentence-transformers here")

    def _boom(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError("an LLM call was paid for a degraded search")

    monkeypatch.setattr(semantic, "_load_embedder", _no_weights)
    monkeypatch.setattr(rank, "expand_query", _boom)
    hits = semantic.search(paths, _QUERY, top_k=5, logger=_LOG)
    assert hits and all(h.degraded for h in hits)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

_FLAG_SETS: list[tuple[str, dict[str, str]]] = [
    ("graph", {"BRAIN_RANK_GRAPH": "1"}),
    ("salience", {"BRAIN_RANK_SALIENCE": "1"}),
    ("rerank", {"BRAIN_RANK_RERANK": "1"}),
    ("expansion", {"BRAIN_QUERY_EXPANSION": "1"}),
    ("all four", {"BRAIN_RANK_GRAPH": "1", "BRAIN_RANK_SALIENCE": "1",
                  "BRAIN_RANK_RERANK": "1", "BRAIN_QUERY_EXPANSION": "1"}),
]


@pytest.mark.parametrize("env", [e for _n, e in _FLAG_SETS],
                         ids=[n for n, _e in _FLAG_SETS])
def test_the_layers_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """Every layer, three runs each, with the process-wide caches dropped
    between runs — so a run cannot come out identical merely because the
    graph, the index or the paraphrase file was already warm, which is all
    the single warm-cache GRAPH+SALIENCE run this replaces could show.
    (Genuine cross-process drift is out of reach from inside one process;
    this is the strongest statement an in-process test can make.)"""
    paths = _vault(tmp_path, monkeypatch)
    _connections(paths, [
        {"a": "caches", "b": "memory", "kind": "cooccurrence", "weight": 3.0,
         "sources": ["uni/arch/lecture8.pdf", "uni/poetry/rhyme.pdf"]},
        {"a": "people/anna", "b": "organisations/acme", "kind": "typed", "weight": 1.0,
         "rel": "works_at", "sources": ["knowledge/people/anna.md"]},
    ])
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(rank, "load_reranker", lambda *a, **k: _ReverseEncoder())
    monkeypatch.setattr(summarize, "generate_structured", _fake_paraphrases)

    runs: list[list[tuple[str, float]]] = []
    for _ in range(3):
        monkeypatch.setattr(rank, "_GRAPH_CACHE", None)
        monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
        monkeypatch.setattr(semantic, "_LEXICAL_CACHE", None)
        runs.append([
            (h.source_relative_path, round(h.score, 12))
            for h in semantic.search(paths, _QUERY, top_k=5, logger=_LOG)
        ])
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# the candidate pool must not change the ranking
# ---------------------------------------------------------------------------

_POOL_QUERY_ROWS = 100    # exactly semantic._CANDIDATES, so top_k=10 fetches these
_POOL_DEEP_ROWS = 40      # reachable only when top_k widens the pool
_POOL_HUB = "deep/hub.pdf"


def _pool_rows() -> list[tuple[str, str, str]]:
    rows = [
        (f"uni/pool/doc{i:03d}.pdf", f"cpu caches and the memory hierarchy part {i}",
         "pdf-mineru")
        for i in range(_POOL_QUERY_ROWS)
    ]
    rows += [
        (f"deep/gardening{j:03d}.pdf", f"gardening tips number {j}", "pdf-mineru")
        for j in range(_POOL_DEEP_ROWS - 1)
    ]
    # Last by dense rank, and the source with the most graph mass: it enters
    # the pool only at a wide top_k.
    rows.append((_POOL_HUB, "gardening notes for the hub", "pdf-mineru"))
    return rows


class _RankedEmbedder:
    """Deterministic by CONSTRUCTION rather than by hash: row ``i`` encodes to
    a vector whose cosine with the query falls with ``i``, so the dense order
    is the row order and the pool at ``top_k=10`` is exactly the first 100
    rows. Anything unknown (the query itself) encodes to the query axis."""

    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self.by_text: dict[str, np.ndarray] = {}
        for position, (_rel, text, _origin) in enumerate(rows):
            vec = np.zeros(_DIM, dtype=np.float32)
            vec[0] = 1.0
            vec[1] = 0.01 * position
            self.by_text[text] = vec / np.linalg.norm(vec)

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        axis = np.zeros(_DIM, dtype=np.float32)
        axis[0] = 1.0
        return np.vstack([self.by_text.get(s, axis) for s in sentences])


def _pool_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VaultPaths:
    rows = _pool_rows()
    paths = paths_for_root(tmp_path)
    paths.ensure()
    embedder = _RankedEmbedder(rows)
    semantic._write_index_files(
        embedder.encode([text for _rel, text, _o in rows]),
        [
            semantic.MetaRow(
                source_relative_path=rel,
                source_hash="h-" + rel,
                title=Path(rel).stem,
                chunk_idx=0,
                text=text,
                origin=origin,
                heading_path="",
                model="BAAI/bge-small-en-v1.5",
                gen="",
            )
            for rel, text, origin in rows
        ],
        vectors_path=paths.metadata / "embeddings.npy",
        meta_path=paths.metadata / "embeddings_meta.jsonl",
    )
    # The hub shares its whole concept set with the ten seed sources, so its
    # raw graph score is the largest in the vault — which is exactly what the
    # old pool-max normalisation divided by once a wide fetch let it in.
    _connections(paths, [
        {"a": "caches", "b": "memory", "kind": "cooccurrence", "weight": 3.0,
         "sources": [f"uni/pool/doc{i:03d}.pdf" for i in range(10)] + [_POOL_HUB]},
    ])
    monkeypatch.setenv("BRAIN_QUERY_INSTRUCTION", "0")
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (embedder, "cpu"))
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
    monkeypatch.setattr(semantic, "_LEXICAL_CACHE", None)
    monkeypatch.setattr(rank, "_GRAPH_CACHE", None)
    return paths


def test_the_graph_boost_does_not_depend_on_how_many_hits_were_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``recency.memory_search`` fetches 500 candidates on a filtered query
    while the eval measures at 10. The boost used to be normalised against the
    max over the CANDIDATE POOL, so widening the fetch rescaled every boost on
    the top ten and the two callers ranked the same query differently — the
    A/B numbers described a ranking the live filtered path never produced."""
    paths = _pool_vault(tmp_path, monkeypatch)
    baseline = {k: semantic.search(paths, _QUERY, top_k=k, logger=_LOG)[:10]
                for k in (10, 50, 500)}
    assert baseline[10] == baseline[50] == baseline[500]   # the fixture itself is stable

    monkeypatch.setenv("BRAIN_RANK_GRAPH", "1")
    boosted = {k: semantic.search(paths, _QUERY, top_k=k, logger=_LOG)[:10]
               for k in (10, 50, 500)}
    assert boosted[10] != baseline[10]                     # the layer really ran
    assert boosted[10] == boosted[50] == boosted[500]
