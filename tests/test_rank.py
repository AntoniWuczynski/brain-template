"""Tests for ingest_lib.rank — the optional, config-gated ranking layers.

Everything here is pure: the graph index is built from literal JSONL lines,
the cross-encoder is a stub, and the query-expansion LLM is monkeypatched.
No model is loaded and no network call is made.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from collections.abc import Mapping
from types import ModuleType

from ingest_lib import rank, summarize
from ingest_lib.config import RankingConfig, VaultPaths, paths_for_root, ranking_config

_LOG = logging.getLogger("test")


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _edge(**fields: object) -> str:
    return json.dumps(fields)


def _cand(row: int, source: str, *, origin: str = "pdf-mineru",
          node_id: str | None = None, text: str = "") -> rank.Candidate:
    return rank.Candidate(row=row, source=source, origin=origin,
                          node_id=node_id, text=text or source)


# ---------------------------------------------------------------------------
# config gating
# ---------------------------------------------------------------------------

def test_ranking_config_defaults_to_every_layer_off() -> None:
    cfg = ranking_config({})
    assert (cfg.graph_boost, cfg.salience, cfg.rerank, cfg.query_expansion) == (
        False, False, False, False,
    )
    assert cfg.any_enabled is False


def test_ranking_config_reads_flags_and_weights() -> None:
    cfg = ranking_config({
        "BRAIN_RANK_GRAPH": "1",
        "BRAIN_RANK_GRAPH_WEIGHT": "0.25",
        "BRAIN_RANK_SALIENCE": "true",
        "BRAIN_RANK_RERANK": "on",
        "BRAIN_RANK_RERANK_TOP_N": "12",
        "BRAIN_QUERY_EXPANSION": "yes",
        "BRAIN_QUERY_EXPANSION_VARIANTS": "99",
    })
    assert cfg.graph_boost and cfg.salience and cfg.rerank and cfg.query_expansion
    assert cfg.graph_weight == 0.25
    assert cfg.rerank_top_n == 12
    # Variants are clamped so a typo cannot fan one search into 99 LLM-driven
    # rankings.
    assert cfg.query_expansion_variants == 5
    assert cfg.any_enabled is True


def test_ranking_config_falls_back_on_an_unparseable_number() -> None:
    cfg = ranking_config({"BRAIN_RANK_GRAPH": "1", "BRAIN_RANK_GRAPH_WEIGHT": "heavy"})
    assert cfg.graph_boost is True
    assert cfg.graph_weight == RankingConfig().graph_weight


def test_ranking_config_treats_an_unknown_flag_value_as_off() -> None:
    assert ranking_config({"BRAIN_RANK_GRAPH": "maybe"}).graph_boost is False
    assert ranking_config({"BRAIN_RANK_GRAPH": "0"}).graph_boost is False


# ---------------------------------------------------------------------------
# connections.jsonl -> GraphIndex
# ---------------------------------------------------------------------------

def test_build_graph_index_maps_sources_to_their_cooccurring_concepts() -> None:
    idx = rank.build_graph_index([
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=2.0,
              sources=["uni/arch/lecture8.pdf", "uni/arch/lecture9.pdf"]),
    ])
    assert idx.concepts_by_source["uni/arch/lecture8.pdf"] == frozenset({"caches", "pipelining"})
    assert idx.concepts_by_source["uni/arch/lecture9.pdf"] == frozenset({"caches", "pipelining"})


def test_build_graph_index_keeps_typed_entity_edges_out_of_the_concept_map() -> None:
    """Typed edges are directional entity relations, not concept co-occurrence;
    mixing them would make an entity id look like a topic of a document."""
    idx = rank.build_graph_index([
        _edge(a="people/anna", b="organisations/acme", kind="typed", weight=1.0,
              rel="works_at", sources=["knowledge/people/anna.md"]),
    ])
    assert idx.concepts_by_source == {}
    assert idx.typed_neighbours["people/anna"] == frozenset({"organisations/acme"})
    assert idx.typed_neighbours["organisations/acme"] == frozenset({"people/anna"})


def test_typed_degree_counts_distinct_neighbours_not_edges() -> None:
    """Supersede-never-delete means one relationship that ended and restarted
    is several edges. Counting edges would score churn as importance."""
    idx = rank.build_graph_index([
        _edge(a="people/anna", b="organisations/acme", kind="typed", weight=1.0,
              rel="works_at", valid_from="2024-01-01", valid_until="2025-01-01"),
        _edge(a="people/anna", b="organisations/acme", kind="typed", weight=1.0,
              rel="works_at", valid_from="2025-06-01", valid_until=""),
    ])
    assert idx.typed_degree("people/anna") == 1
    assert idx.typed_degree("people/nobody") == 0


def test_build_graph_index_skips_malformed_lines() -> None:
    idx = rank.build_graph_index([
        "not json at all",
        "[1, 2, 3]",
        _edge(a="caches", kind="cooccurrence", weight=1.0),           # no b
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight="lots"),
        "",
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=1.0,
              sources=["uni/arch/lecture8.pdf"]),
    ])
    assert idx.concepts_by_source == {"uni/arch/lecture8.pdf": frozenset({"caches", "pipelining"})}


def test_load_graph_returns_an_empty_index_when_the_file_is_absent(tmp_path: Path) -> None:
    idx = rank.load_graph(_vault(tmp_path), _LOG)
    assert idx.is_empty
    assert rank.graph_affinity([_cand(0, "a.pdf")], idx) == {}


def test_load_graph_reads_and_caches_the_live_file(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (paths.metadata / "connections.jsonl").write_text(
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=1.0,
              sources=["uni/arch/lecture8.pdf"]) + "\n",
        encoding="utf-8",
    )
    first = rank.load_graph(paths, _LOG)
    assert first.concepts_by_source["uni/arch/lecture8.pdf"] == frozenset({"caches", "pipelining"})
    assert rank.load_graph(paths, _LOG) is first   # cached by mtime+size


# ---------------------------------------------------------------------------
# graph adjacency
# ---------------------------------------------------------------------------

def test_graph_affinity_favours_a_hit_sharing_concepts_with_the_top_hits() -> None:
    idx = rank.build_graph_index([
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=3.0,
              sources=["seed.pdf", "adjacent.pdf"]),
        _edge(a="poetry", b="rhyme", kind="cooccurrence", weight=3.0,
              sources=["unrelated.pdf"]),
    ])
    ordered = [_cand(0, "seed.pdf"), _cand(1, "unrelated.pdf"), _cand(2, "adjacent.pdf")]
    aff = rank.graph_affinity(ordered, idx)
    assert aff[2] > aff[1]
    assert all(0.0 <= v <= 1.0 for v in aff.values())


def test_graph_affinity_does_not_credit_a_seed_for_its_own_concepts() -> None:
    """Leave-one-out: the signal is adjacency to the OTHER top hits. A source
    that shares nothing with anything else scores zero however many topics it
    carries, so the boost cannot simply re-state the fused rank."""
    idx = rank.build_graph_index([
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=3.0,
              sources=["lonely.pdf"]),
    ])
    aff = rank.graph_affinity([_cand(0, "lonely.pdf")], idx)
    assert aff[0] == pytest.approx(0.0)


def test_graph_affinity_credits_a_typed_neighbour_of_a_seed_entity() -> None:
    idx = rank.build_graph_index([
        _edge(a="people/anna", b="organisations/acme", kind="typed", weight=1.0,
              rel="works_at", sources=["knowledge/people/anna.md"]),
    ])
    ordered = [
        _cand(0, "knowledge/people/anna.md", origin="knowledge-note", node_id="people/anna"),
        _cand(1, "knowledge/people/zed.md", origin="knowledge-note", node_id="people/zed"),
        _cand(2, "knowledge/organisations/acme.md", origin="knowledge-note",
              node_id="organisations/acme"),
    ]
    aff = rank.graph_affinity(ordered, idx)
    assert aff[2] > aff[1]


def test_graph_affinity_expands_exactly_one_hop() -> None:
    """A concept two hops from the query's profile earns nothing: expansion
    runs over a snapshot, so an expanded concept cannot seed another hop."""
    idx = rank.build_graph_index([
        _edge(a="alpha", b="alpha2", kind="cooccurrence", weight=2.0, sources=["seed.pdf"]),
        _edge(a="alpha", b="beta", kind="semantic", weight=0.9),
        _edge(a="beta", b="gamma", kind="semantic", weight=0.9),
        _edge(a="beta", b="beta2", kind="cooccurrence", weight=2.0, sources=["one-hop.pdf"]),
        _edge(a="gamma", b="gamma2", kind="cooccurrence", weight=2.0, sources=["two-hop.pdf"]),
    ])
    ordered = [_cand(0, "seed.pdf"), _cand(1, "one-hop.pdf"), _cand(2, "two-hop.pdf")]
    # seeds=1: only the top hit defines the profile, so "one-hop" and
    # "two-hop" are genuinely one and two hops from it.
    aff = rank.graph_affinity(ordered, idx, seeds=1)
    assert aff[1] > aff[2]
    assert aff[2] == pytest.approx(0.0)


def test_graph_affinity_seeds_only_from_the_top_of_the_fused_list() -> None:
    """The profile decides every boost, so it has to be a function of the query
    alone. Collecting ten DISTINCT sources without a row bound reached fused
    rank 34 (median) on the live index, where the order moves as soon as the
    candidate pool widens — which made the boosted top ten a function of the
    caller's ``top_k``."""
    idx = rank.build_graph_index([
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=3.0,
              sources=["seed.pdf"]),
        _edge(a="poetry", b="rhyme", kind="cooccurrence", weight=3.0,
              sources=["deep.pdf", "also-poetry.pdf"]),
    ])
    # One source, many chunks — the shape that makes the scan walk deep.
    ordered = [_cand(i, "seed.pdf") for i in range(rank._SEED_SCAN)]
    ordered += [
        _cand(rank._SEED_SCAN, "deep.pdf"),
        _cand(rank._SEED_SCAN + 1, "also-poetry.pdf"),
    ]
    aff = rank.graph_affinity(ordered, idx)
    # "also-poetry.pdf" shares its whole concept set with "deep.pdf" and
    # nothing with the top hit, so it scores only if a source from below the
    # scan window was allowed to seed the profile.
    assert aff[rank._SEED_SCAN + 1] == pytest.approx(0.0)


def test_graph_boosts_are_bounded_by_weight_times_the_unit_and_never_negative() -> None:
    idx = rank.build_graph_index([
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=3.0,
              sources=["seed.pdf", "adjacent.pdf"]),
    ])
    ordered = [_cand(0, "seed.pdf"), _cand(1, "adjacent.pdf"), _cand(2, "elsewhere.pdf")]
    unit = 1.0 / 61.0
    boosts = rank.graph_boosts(ordered, idx, unit=unit, weight=0.5)
    assert boosts
    assert all(0.0 <= b <= 0.5 * unit + 1e-12 for b in boosts.values())


def test_graph_boosts_are_empty_at_zero_weight() -> None:
    idx = rank.build_graph_index([
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=3.0,
              sources=["seed.pdf"]),
    ])
    assert rank.graph_boosts([_cand(0, "seed.pdf")], idx, unit=0.016, weight=0.0) == {}


# ---------------------------------------------------------------------------
# salience
# ---------------------------------------------------------------------------

def test_salience_scores_an_archive_chunk_zero() -> None:
    idx = rank.build_graph_index([])
    assert rank.salience(_cand(0, "uni/arch/lecture8.pdf"), idx) == 0.0


def test_salience_rewards_a_curated_note_and_saturates_with_typed_degree() -> None:
    idx = rank.build_graph_index([
        _edge(a="people/anna", b=f"organisations/o{i}", kind="typed", weight=1.0, rel="works_at")
        for i in range(40)
    ])
    plain = rank.salience(
        _cand(0, "knowledge/notes/x.md", origin="knowledge-note"), idx
    )
    hub = rank.salience(
        _cand(1, "knowledge/people/anna.md", origin="knowledge-note", node_id="people/anna"), idx
    )
    assert plain == pytest.approx(0.5)
    assert hub == pytest.approx(1.0)   # saturated, not unbounded


def test_salience_boosts_only_ever_add(tmp_path: Path) -> None:
    """The archive must not be pushed DOWN in absolute terms: an archive
    chunk's boost is exactly zero, never negative."""
    idx = rank.build_graph_index([])
    boosts = rank.salience_boosts(
        [_cand(0, "uni/x.pdf"), _cand(1, "knowledge/notes/y.md", origin="knowledge-note")],
        idx, unit=1.0 / 61.0, weight=0.5,
    )
    assert boosts[0] == 0.0
    assert boosts[1] > 0.0


# ---------------------------------------------------------------------------
# cross-encoder rerank
# ---------------------------------------------------------------------------

class _StubEncoder:
    """Structural, NOT a subclass: ``rank.CrossEncoderModel`` is a Protocol,
    so a double that merely has ``predict`` type-checks — exactly as the real
    ``sentence_transformers.CrossEncoder``, which subclasses nothing here,
    has to."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]:
        self.calls.append(sentences)
        return self.scores


def test_rerank_rows_orders_by_score_and_breaks_ties_on_fused_position() -> None:
    window = [_cand(7, "a.pdf"), _cand(3, "b.pdf"), _cand(9, "c.pdf")]
    model = _StubEncoder([0.1, 5.0, 5.0])
    assert rank.rerank_rows("q", window, model, _LOG) == [3, 9, 7]
    assert model.calls == [[("q", "a.pdf"), ("q", "b.pdf"), ("q", "c.pdf")]]


def test_rerank_rows_refuses_a_length_mismatch() -> None:
    window = [_cand(0, "a.pdf"), _cand(1, "b.pdf")]
    assert rank.rerank_rows("q", window, _StubEncoder([1.0]), _LOG) is None


def test_rerank_rows_survives_a_scoring_failure() -> None:
    class _Boom:
        def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]:
            raise RuntimeError("no torch here")

    assert rank.rerank_rows("q", [_cand(0, "a.pdf")], _Boom(), _LOG) is None


def test_load_reranker_returns_none_when_the_package_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI installs neither torch nor sentence-transformers; the layer has to
    degrade to the unmodified fused order rather than raise."""
    import builtins

    real_import = builtins.__import__

    def _no_sentence_transformers(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "sentence_transformers":
            raise ImportError("no module named sentence_transformers")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_sentence_transformers)
    monkeypatch.setattr(rank, "_RERANKER_CACHE", {})
    assert rank.load_reranker("cross-encoder/whatever", _LOG) is None


# ---------------------------------------------------------------------------
# query expansion
# ---------------------------------------------------------------------------

def _expansion_cfg() -> RankingConfig:
    return ranking_config({"BRAIN_QUERY_EXPANSION": "1"})


def test_expand_query_returns_nothing_without_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(summarize, "generate_structured", lambda **kwargs: None)
    assert rank.expand_query(_vault(tmp_path), "hash tables", _expansion_cfg(), _LOG) == []


def test_expand_query_caches_by_query_so_an_eval_run_pays_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    calls: list[str] = []

    def _fake(**kwargs: object) -> object:
        user = kwargs["user"]
        assert isinstance(user, str)
        calls.append(user)
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["associative arrays", "constant-time lookup structure"])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    first = rank.expand_query(paths, "hash tables", _expansion_cfg(), _LOG)
    second = rank.expand_query(paths, "hash tables", _expansion_cfg(), _LOG)
    assert first == ["associative arrays", "constant-time lookup structure"]
    assert second == first
    assert len(calls) == 1
    assert (paths.metadata / ".cache" / "query-expansion.json").is_file()


def test_expand_query_drops_the_original_query_and_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake(**kwargs: object) -> object:
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["Hash Tables", "associative arrays", "associative arrays", " "])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    got = rank.expand_query(_vault(tmp_path), "hash tables", _expansion_cfg(), _LOG)
    assert got == ["associative arrays"]


def test_expand_query_ignores_an_unreadable_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    cache = paths.metadata / ".cache" / "query-expansion.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{not json", encoding="utf-8")

    def _fake(**kwargs: object) -> object:
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["associative arrays"])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    assert rank.expand_query(paths, "hash tables", _expansion_cfg(), _LOG) == ["associative arrays"]


def test_expand_query_re_asks_when_the_prompt_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache keyed on the query alone served the OLD prompt's paraphrases
    forever: every A/B after an edit silently re-measured a prompt that no
    longer existed, and only a human remembering to delete the file recovered."""
    paths = _vault(tmp_path)
    calls: list[str] = []

    def _fake(**kwargs: object) -> object:
        system = kwargs["system"]
        assert isinstance(system, str)
        calls.append(system)
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["associative arrays", "key-value lookup"])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    cfg = _expansion_cfg()
    rank.expand_query(paths, "hash tables", cfg, _LOG)
    rank.expand_query(paths, "hash tables", cfg, _LOG)
    assert len(calls) == 1
    monkeypatch.setattr(
        rank, "_EXPANSION_SYSTEM", rank._EXPANSION_SYSTEM + " Prefer single nouns."
    )
    rank.expand_query(paths, "hash tables", cfg, _LOG)
    assert len(calls) == 2
    assert calls[1].endswith("Prefer single nouns.")


def test_expand_query_re_asks_when_the_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching provider or model changes the answer, so it has to change the
    key: otherwise the new model is never asked and the eval measures the old."""
    paths = _vault(tmp_path)
    calls: list[str] = []

    def _fake(**kwargs: object) -> object:
        user = kwargs["user"]
        assert isinstance(user, str)
        calls.append(user)
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["associative arrays"])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("BRAIN_LLM_MODEL", "claude-haiku-4-5")
    cfg = _expansion_cfg()
    rank.expand_query(paths, "hash tables", cfg, _LOG)
    rank.expand_query(paths, "hash tables", cfg, _LOG)
    assert len(calls) == 1
    monkeypatch.setenv("BRAIN_LLM_MODEL", "claude-sonnet-5")
    rank.expand_query(paths, "hash tables", cfg, _LOG)
    assert len(calls) == 2


def test_expand_query_re_asks_when_more_variants_are_wanted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BRAIN_QUERY_EXPANSION_VARIANTS was inert upward: the documented 1-5 knob
    could never exceed whatever the first call happened to cache."""
    paths = _vault(tmp_path)
    calls: list[int] = []

    def _fake(**kwargs: object) -> object:
        calls.append(1)
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["p1", "p2", "p3", "p4", "p5"])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    three = rank.expand_query(paths, "hash tables", _expansion_cfg(), _LOG)
    assert len(three) == 3 and len(calls) == 1
    five = rank.expand_query(
        paths,
        "hash tables",
        ranking_config({"BRAIN_QUERY_EXPANSION": "1", "BRAIN_QUERY_EXPANSION_VARIANTS": "5"}),
        _LOG,
    )
    assert len(five) == 5
    assert len(calls) == 2


def test_expansion_cache_evicts_the_oldest_entry_not_the_first_alphabetically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is written ``sort_keys=True``, so the order a reload sees is
    alphabetical. Evicting "the front" of it dropped whichever query sorted
    first — as likely the newest entry as the oldest — and that query then
    re-paid an LLM call on every search while late-sorting ones were immortal."""
    paths = _vault(tmp_path)
    calls: list[int] = []

    def _fake(**kwargs: object) -> object:
        calls.append(1)
        schema = kwargs["schema"]
        assert isinstance(schema, type)
        return schema(paraphrases=["one", "two", "three"])

    monkeypatch.setattr(summarize, "generate_structured", _fake)
    monkeypatch.setattr(rank, "_MAX_CACHED_EXPANSIONS", 3)
    cfg = _expansion_cfg()
    for query in ("zzz oldest", "mmm middle", "aaa newest"):
        rank.expand_query(paths, query, cfg, _LOG)
    assert len(calls) == 3
    rank.expand_query(paths, "bbb brand new", cfg, _LOG)   # over the cap
    assert len(calls) == 4
    rank.expand_query(paths, "aaa newest", cfg, _LOG)      # newest: still cached
    assert len(calls) == 4
    rank.expand_query(paths, "zzz oldest", cfg, _LOG)      # oldest: the evicted one
    assert len(calls) == 5


# ---------------------------------------------------------------------------
# process-wide caches under concurrency
# ---------------------------------------------------------------------------

def test_load_graph_parses_the_file_once_under_concurrent_searches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mcp_server/app.py`` runs searches in a thread pool, so two cold
    searches used to parse the whole connections file each."""
    paths = _vault(tmp_path)
    (paths.metadata / "connections.jsonl").write_text(
        _edge(a="caches", b="pipelining", kind="cooccurrence", weight=1.0,
              sources=["uni/arch/lecture8.pdf"]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rank, "_GRAPH_CACHE", None)
    parses: list[int] = []
    real_build = rank.build_graph_index

    def _slow_build(lines: Sequence[str]) -> rank.GraphIndex:
        parses.append(1)
        time.sleep(0.1)
        return real_build(lines)

    monkeypatch.setattr(rank, "build_graph_index", _slow_build)
    ready = threading.Barrier(2)

    def _search() -> None:
        ready.wait()
        rank.load_graph(paths, _LOG)

    threads = [threading.Thread(target=_search) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(parses) == 1


_RERANKER_LOADS: list[str] = []


class _SlowCrossEncoder:
    """Stands in for ``sentence_transformers.CrossEncoder``: records each load
    and takes long enough that the second thread is certainly inside
    ``load_reranker`` while the first is still loading."""

    def __init__(self, model_name: str) -> None:
        _RERANKER_LOADS.append(model_name)
        time.sleep(0.1)

    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]:
        return [0.0] * len(sentences)


class _FakeSentenceTransformers(ModuleType):
    """The optional package, faked into ``sys.modules`` so ``load_reranker``'s
    lazy import finds it."""

    CrossEncoder: type[_SlowCrossEncoder]

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.CrossEncoder = _SlowCrossEncoder


def test_load_reranker_loads_the_weights_once_under_concurrent_searches(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cold load is ~20 s; without a lock two concurrent first searches
    paid it twice and raced to fill the cache."""
    _RERANKER_LOADS.clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeSentenceTransformers())
    monkeypatch.setattr(rank, "_RERANKER_CACHE", {})
    ready = threading.Barrier(2)

    def _search() -> None:
        ready.wait()
        rank.load_reranker("cross-encoder/whatever", _LOG)

    threads = [threading.Thread(target=_search) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _RERANKER_LOADS == ["cross-encoder/whatever"]
