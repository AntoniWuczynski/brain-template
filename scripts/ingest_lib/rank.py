"""Optional ranking layers over the hybrid fusion in ``semantic.search``.

Four independent, config-gated layers. **All are off by default** (see
``config.ranking_config``): with no environment set nothing here runs and the
fused ranking is byte-for-byte what it was before this module existed. Each is
switched separately so ``scripts/eval_retrieval.py --compare`` can score one at
a time against the golden set.

1. **Graph adjacency** (``BRAIN_RANK_GRAPH``) — a bounded additive boost read
   from ``metadata/connections.jsonl``. Two signals, deliberately kept apart
   because the file mixes three edge kinds that mean different things:

   - *co-occurrence* edges carry the SOURCES that back them, so they give a
     source -> concept membership map for free. A hit is boosted when its
     source's concepts overlap the concept profile of the query's own top
     hits (one hop of expansion through the concept graph), and again when
     its source is *corroborated* — sharing concepts with several distinct
     other sources the query already surfaced.
   - *typed* edges are directional entity relations (``people/x works_at
     organisations/y``). They connect entity NOTES, not concepts, so they are
     used only to relate a knowledge-note hit to the entity nodes the query's
     top hits already name.

   *semantic* concept edges (centroid cosine) are used for the one-hop
   expansion only — they are derived from the same embeddings the dense
   ranking already used, so counting them as independent evidence would
   double-count the dense signal.

   Cost is O(hits x concepts-per-source x neighbour-cap), not O(graph): the
   file is parsed once per generation and cached.

2. **Salience** (``BRAIN_RANK_SALIENCE``) — a small additive weight from a
   note's OWN signals: its typed-relation degree, and whether it is a curated
   ``knowledge/`` note rather than a raw archive chunk. It only ever ADDS to
   curated notes, so an archive hit is never pushed down in absolute terms —
   but it does change relative order, which is exactly what the eval measures.

3. **Cross-encoder rerank** (``BRAIN_RANK_RERANK``) — reorders the top
   ``BRAIN_RANK_RERANK_TOP_N`` fused hits with ``cross-encoder/
   ms-marco-MiniLM-L-6-v2`` (22.7M params, ~90 MB of weights, Apache-2.0),
   loaded through the ``sentence_transformers`` package the embedder already
   needs. Imported lazily behind ``try/except`` like the embedder, so the
   locked CI environment (which installs neither torch nor
   sentence-transformers) is unaffected. The window's existing fused scores
   are re-assigned in the new order rather than replaced by cross-encoder
   logits, so ``SearchHit.score`` keeps its documented RRF scale and
   ``recency.memory_search``'s multiplication stays meaningful.

4. **Query expansion** (``BRAIN_QUERY_EXPANSION``) — asks the configured LLM
   router for 2-3 paraphrases, whose rankings are fused in alongside the
   original query's at a discount. Off by default because it costs an LLM
   call per search; paraphrases are cached in
   ``metadata/.cache/query-expansion.json`` (a regenerable cache, under the
   ``.cache/`` name .gitignore already excludes) so an eval run pays once and
   is deterministic thereafter. The key is the query plus a digest of the
   prompt, the variant count and the provider/model, so an edited prompt
   re-asks instead of serving the previous prompt's answer.

Determinism: no randomness anywhere. Every ordering here breaks ties on a
stable key (the candidate's row index), and every accumulation runs over a
sorted iteration order.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING, Protocol

from .atomic import atomic_write_text
from .config import RankingConfig, VaultPaths
from .knowledge import KNOWLEDGE_EXTRACTOR

if TYPE_CHECKING:
    # semantic imports this module, so its MetaRow is a type-only import.
    from .semantic import MetaRow

# --- tunables (not env-exposed: the weights are the knobs that matter) ------
_SEED_SOURCES = 10      # how many distinct top sources define the query's profile
_SEED_SCAN = 10         # ... and how far down the fused list they may be collected from
_NEIGHBOUR_CAP = 8      # graph neighbours kept per concept (mirrors _RELATED_TOP_N)
_HOP_DECAY = 0.5        # weight of a one-hop-expanded concept vs a seed concept
_CORROBORATION_SHARE = 0.3   # split between profile affinity and corroboration
_DEGREE_SATURATION = 10.0    # typed-degree at which the salience term maxes out
_MAX_CACHED_EXPANSIONS = 2000

_KIND_COOCCURRENCE = "cooccurrence"
_KIND_SEMANTIC = "semantic"
_KIND_TYPED = "typed"

_EXPANSION_CACHE_DIR = ".cache"
_EXPANSION_CACHE_FILE = "query-expansion.json"


@dataclass(frozen=True)
class Candidate:
    """One fused hit, reduced to what the ranking layers need.

    ``node_id`` is the graph node this source IS, resolved by the caller
    through ``relations.canonical_entity_id`` — never by testing the path
    prefix, which is not the vault's definition of entity identity.
    """

    row: int                 # index into the search's meta rows; the tie-break key
    source: str              # source_relative_path
    origin: str              # the record's extractor ("knowledge-note", "pdf-mineru", ...)
    node_id: str | None      # entity node id when the source is one, else None
    text: str                # the chunk text (reranker input)


@dataclass(frozen=True)
class GraphIndex:
    """``metadata/connections.jsonl``, indexed for per-hit lookup.

    ``concepts_by_source`` comes from co-occurrence edges (the only kind that
    records its backing sources), so a source whose summary carried fewer than
    two topics has no entry — it simply gets no graph signal, which is the
    honest answer rather than a fabricated one.
    """

    concepts_by_source: Mapping[str, frozenset[str]] = field(default_factory=dict)
    neighbours: Mapping[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    typed_neighbours: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.concepts_by_source and not self.typed_neighbours

    def typed_degree(self, node_id: str) -> int:
        """Distinct typed neighbours, NOT edge count: supersede-never-delete
        means one relationship that ended and restarted is several edges, and
        counting those would score churn as importance."""
        return len(self.typed_neighbours.get(node_id, frozenset()))


# ---------------------------------------------------------------------------
# connections.jsonl -> GraphIndex
# ---------------------------------------------------------------------------

_GraphKey = tuple[str, float, int]
_GRAPH_CACHE: tuple[_GraphKey, GraphIndex] | None = None
# Held only while PARSING the file, never while reading the cache — the same
# split ``semantic._LEXICAL_BUILD_LOCK`` uses, and for the same reason: the
# MCP server runs searches in a thread pool (``mcp_server/app.py``), so
# without it a cold generation is parsed once per concurrent search.
_GRAPH_LOCK = threading.Lock()


def _edge_from_json(obj: object) -> tuple[str, str, str, float, tuple[str, ...]] | None:
    """``(a, b, kind, weight, sources)`` for one well-formed edge line, else None.

    The file is written by ``connections.py``, but it is a plain-text vault
    artefact a human can hand-edit, so a malformed line is skipped rather than
    crashing every search that follows.
    """
    if not isinstance(obj, dict):
        return None
    a = obj.get("a")
    b = obj.get("b")
    kind = obj.get("kind")
    weight = obj.get("weight")
    if not isinstance(a, str) or not isinstance(b, str) or not isinstance(kind, str):
        return None
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        return None
    raw_sources = obj.get("sources")
    sources = (
        tuple(s for s in raw_sources if isinstance(s, str))
        if isinstance(raw_sources, list)
        else ()
    )
    return a, b, kind, float(weight), sources


def build_graph_index(lines: Sequence[str]) -> GraphIndex:
    """Index already-read ``connections.jsonl`` lines. Pure; unit-testable."""
    concepts: dict[str, set[str]] = {}
    raw_neighbours: dict[str, dict[str, float]] = {}
    typed: dict[str, set[str]] = {}
    co_max: dict[str, float] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except ValueError:
            continue
        edge = _edge_from_json(decoded)
        if edge is None:
            continue
        a, b, kind, weight, sources = edge
        if kind == _KIND_TYPED:
            typed.setdefault(a, set()).add(b)
            typed.setdefault(b, set()).add(a)
            continue
        if kind == _KIND_COOCCURRENCE:
            for src in sources:
                bucket = concepts.setdefault(src, set())
                bucket.add(a)
                bucket.add(b)
            co_max[a] = max(co_max.get(a, 0.0), weight)
            co_max[b] = max(co_max.get(b, 0.0), weight)
            raw_neighbours.setdefault(a, {})[b] = weight
            raw_neighbours.setdefault(b, {})[a] = weight
        elif kind == _KIND_SEMANTIC:
            # Cosine is already in [0, 1]; keep the stronger signal if a pair
            # is linked by both kinds (co-occurrence is normalised below).
            raw_neighbours.setdefault(a, {}).setdefault(b, 0.0)
            raw_neighbours.setdefault(b, {}).setdefault(a, 0.0)
            raw_neighbours[a][b] = max(raw_neighbours[a][b], weight)
            raw_neighbours[b][a] = max(raw_neighbours[b][a], weight)

    neighbours: dict[str, tuple[tuple[str, float], ...]] = {}
    for concept, nbrs in raw_neighbours.items():
        scale = co_max.get(concept, 0.0)
        scored = [
            (other, w / scale if scale > 0 and w > 1.0 else min(w, 1.0))
            for other, w in nbrs.items()
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        neighbours[concept] = tuple(scored[:_NEIGHBOUR_CAP])

    return GraphIndex(
        concepts_by_source={s: frozenset(c) for s, c in concepts.items()},
        neighbours=neighbours,
        typed_neighbours={n: frozenset(v) for n, v in typed.items()},
    )


def load_graph(paths: VaultPaths, logger: logging.Logger | None = None) -> GraphIndex:
    """``metadata/connections.jsonl`` as a ``GraphIndex``, cached by mtime+size.

    An absent or unreadable file yields an EMPTY index, which makes every
    boost zero — the layer degrades to the unchanged baseline instead of
    failing the search.
    """
    global _GRAPH_CACHE
    log = logger or logging.getLogger(__name__)
    path = paths.metadata / "connections.jsonl"
    try:
        stat = path.stat()
    except OSError:
        return GraphIndex()
    key: _GraphKey = (str(path), stat.st_mtime, stat.st_size)
    cache = _GRAPH_CACHE
    if cache is not None and cache[0] == key:
        return cache[1]
    with _GRAPH_LOCK:
        # Re-check: another thread may have parsed this generation while this
        # one waited for the lock.
        cache = _GRAPH_CACHE
        if cache is not None and cache[0] == key:
            return cache[1]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("rank: could not read %s (%s) — graph boost inactive", path, exc)
            return GraphIndex()
        index = build_graph_index(text.splitlines())
        _GRAPH_CACHE = (key, index)
    return index


# ---------------------------------------------------------------------------
# graph adjacency boost
# ---------------------------------------------------------------------------

def graph_affinity(
    ordered: Sequence[Candidate], graph: GraphIndex, *, seeds: int = _SEED_SOURCES
) -> dict[int, float]:
    """Per-candidate graph affinity in ``[0, 1]``, keyed by ``Candidate.row``.

    ``ordered`` is the fused candidate pool in rank order; the distinct
    sources among its first ``_SEED_SCAN`` entries (at most ``seeds`` of them)
    define the query's concept/entity profile (weight 1/rank), which is
    expanded one hop through the concept graph. Returns an empty mapping when
    the graph carries no usable signal.

    The scan is bounded by ROWS, not just by distinct sources, because the
    profile decides every boost and therefore has to be a function of the
    query alone. Collecting ten distinct sources takes a median of 34 fused
    rows on the live index, and the fused order that deep MOVES when the
    candidate pool widens (measured over 30 golden queries: the top 10 rows
    differ between a 100- and a 500-candidate pool for 2 queries, the first
    30 rows for 22) — so an unbounded scan made the profile, and with it the
    ranking of the top ten, a function of the caller's ``top_k``.
    """
    if graph.is_empty or not ordered:
        return {}

    seed_sources: list[str] = []
    seed_nodes: list[str | None] = []
    for cand in ordered[:_SEED_SCAN]:
        if cand.source in seed_sources:
            continue
        seed_sources.append(cand.source)
        seed_nodes.append(cand.node_id)
        if len(seed_sources) >= seeds:
            break

    profile: dict[str, float] = {}
    node_profile: dict[str, float] = {}
    seed_concepts: list[frozenset[str]] = []
    # What each seed contributed, so a seed's affinity can leave its OWN
    # contribution out: the signal is "adjacent to the other top hits", and
    # without this every seed would score its own concepts back and the boost
    # would partly just re-state the fused rank it already has.
    own_concepts: dict[str, dict[str, float]] = {}
    own_nodes: dict[str, dict[str, float]] = {}
    for position, (src, node) in enumerate(
        zip(seed_sources, seed_nodes, strict=True), start=1
    ):
        w = 1.0 / position
        concepts = graph.concepts_by_source.get(src, frozenset())
        seed_concepts.append(concepts)
        mine = own_concepts.setdefault(src, {})
        for concept in sorted(concepts):
            profile[concept] = profile.get(concept, 0.0) + w
            mine[concept] = mine.get(concept, 0.0) + w
        if node is not None:
            my_nodes = own_nodes.setdefault(src, {})
            node_profile[node] = node_profile.get(node, 0.0) + w
            my_nodes[node] = my_nodes.get(node, 0.0) + w
            for neighbour in sorted(graph.typed_neighbours.get(node, frozenset())):
                node_profile[neighbour] = node_profile.get(neighbour, 0.0) + w * _HOP_DECAY
                my_nodes[neighbour] = my_nodes.get(neighbour, 0.0) + w * _HOP_DECAY

    # One hop of expansion, applied to a snapshot so an expanded concept can
    # never seed a second hop (that would be an unbounded random walk).
    expanded = dict(profile)
    for concept, w in sorted(profile.items()):
        for other, strength in graph.neighbours.get(concept, ()):
            expanded[other] = expanded.get(other, 0.0) + w * strength * _HOP_DECAY
    for mine in own_concepts.values():
        for concept, w in sorted(list(mine.items())):
            for other, strength in graph.neighbours.get(concept, ()):
                mine[other] = mine.get(other, 0.0) + w * strength * _HOP_DECAY

    raw: dict[int, float] = {}
    corroboration: dict[int, float] = {}
    for cand in ordered:
        concepts = graph.concepts_by_source.get(cand.source, frozenset())
        mine = own_concepts.get(cand.source, {})
        if concepts:
            # sqrt-normalised: a source tagged with 20 topics should not
            # out-score a precise one purely by having more mass to sum.
            total = sum(
                max(0.0, expanded.get(c, 0.0) - mine.get(c, 0.0)) for c in concepts
            )
            score = total / math.sqrt(len(concepts))
        else:
            score = 0.0
        if cand.node_id is not None:
            score += max(
                0.0,
                node_profile.get(cand.node_id, 0.0)
                - own_nodes.get(cand.source, {}).get(cand.node_id, 0.0),
            )
        raw[cand.row] = score
        shared = sum(
            1
            for src, seed_set in zip(seed_sources, seed_concepts, strict=True)
            if src != cand.source and seed_set & concepts
        )
        corroboration[cand.row] = shared / len(seed_sources) if seed_sources else 0.0

    # Normalise against the SEED candidates, not against the candidate pool.
    # Dividing by ``max(raw.values())`` made every boost a function of how
    # many candidates were fetched — the pool is
    # ``min(n, max(_CANDIDATES, top_k))`` (semantic.search) — so widening the
    # fetch rescaled the boosts on the top ten, and
    # ``recency.memory_search``'s filtered path (fetch 500) ranked a query
    # differently from a plain ``search(top_k=10)``, which is the ranking the
    # A/B measured. The seed rows are a fixed prefix, so this reference is the
    # same at every pool size, and it keeps the original meaning: affinity 1.0
    # is "as adjacent to the top hits as the best-connected top hit itself".
    # Clamped, because a candidate below the seeds can exceed that; zero when
    # no seed is adjacent to any other, which leaves corroboration alone (9 of
    # the 87 golden queries reach at most that ceiling).
    seed_rows = [c.row for c in ordered[:_SEED_SCAN]]
    peak = max((raw[r] for r in seed_rows), default=0.0)
    if peak <= 0.0:
        return {row: _CORROBORATION_SHARE * corroboration[row] for row in raw}
    return {
        row: (1.0 - _CORROBORATION_SHARE) * min(1.0, value / peak)
        + _CORROBORATION_SHARE * corroboration[row]
        for row, value in raw.items()
    }


def graph_boosts(
    ordered: Sequence[Candidate],
    graph: GraphIndex,
    *,
    unit: float,
    weight: float,
) -> dict[int, float]:
    """Bounded additive boosts on the RRF scale: at most ``weight * unit``."""
    if weight <= 0.0:
        return {}
    return {
        row: weight * unit * affinity
        for row, affinity in graph_affinity(ordered, graph).items()
    }


# ---------------------------------------------------------------------------
# salience
# ---------------------------------------------------------------------------

_KNOWLEDGE_ORIGIN = "knowledge-note"


def salience(cand: Candidate, graph: GraphIndex) -> float:
    """A note's own importance in ``[0, 1]``, independent of the query.

    Half of it is "this is a curated knowledge note, not a raw archive
    chunk"; half is the entity's typed-relation degree, log-saturated so a
    hub with 40 relations does not out-rank everything else forever. An
    archive chunk scores 0 — it is never pushed DOWN, only not pushed up.
    """
    curated = 1.0 if cand.origin == _KNOWLEDGE_ORIGIN else 0.0
    degree = graph.typed_degree(cand.node_id) if cand.node_id is not None else 0
    degree_term = min(1.0, math.log1p(degree) / math.log1p(_DEGREE_SATURATION))
    return 0.5 * curated + 0.5 * degree_term


def salience_boosts(
    ordered: Sequence[Candidate], graph: GraphIndex, *, unit: float, weight: float
) -> dict[int, float]:
    if weight <= 0.0:
        return {}
    return {c.row: weight * unit * salience(c, graph) for c in ordered}


# ---------------------------------------------------------------------------
# cross-encoder rerank
# ---------------------------------------------------------------------------

class CrossEncoderModel(Protocol):
    """Structural stand-in for ``sentence_transformers.CrossEncoder``.

    Only ``predict`` is called. Declared here (rather than importing the real
    class for typing) because the package is deliberately absent from the
    locked CI environment — and declared as a ``Protocol``, like
    ``semantic.Embedder``, because the real ``CrossEncoder`` is not a subclass
    of anything in this repo: a nominal class here would make the annotations
    on ``_RERANKER_CACHE`` and ``rerank_rows`` true only of the test doubles.
    """

    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]: ...


_RERANKER_CACHE: dict[str, CrossEncoderModel] = {}
# Serialises the ~20 s cold load so concurrent MCP searches load the weights
# once between them rather than once each (see _GRAPH_LOCK).
_RERANKER_LOCK = threading.Lock()


def load_reranker(model_name: str, logger: logging.Logger) -> CrossEncoderModel | None:
    """Lazily load (and cache) the cross-encoder, or None if unavailable.

    ~90 MB of weights, downloaded from Hugging Face on first use, then cached
    under ``~/.cache/huggingface``. Any failure — package missing, no network,
    a model name that does not resolve — logs and returns None so the search
    falls back to the unmodified fused order.
    """
    cached = _RERANKER_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _RERANKER_LOCK:
        cached = _RERANKER_CACHE.get(model_name)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # noqa: BLE001 — an optional extra, absent by design in CI
            logger.warning("rank: cross-encoder unavailable (%r) — rerank skipped", exc)
            return None
        try:
            model: CrossEncoderModel = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 — download/load can fail many ways
            logger.warning("rank: cross-encoder %s failed to load (%r)", model_name, exc)
            return None
        _RERANKER_CACHE[model_name] = model
    return model


def rerank_rows(
    query: str,
    window: Sequence[Candidate],
    model: CrossEncoderModel,
    logger: logging.Logger,
) -> list[int] | None:
    """The window's rows, reordered by cross-encoder relevance.

    Ties break on the incoming fused position, so the result is a
    deterministic function of (query, window). None on a scoring failure or a
    length mismatch, which leaves the caller's order untouched.
    """
    if not window:
        return []
    try:
        scores = model.predict([(query, c.text) for c in window])
    except Exception as exc:  # noqa: BLE001 — inference can fail many ways
        logger.warning("rank: cross-encoder scoring failed (%r) — order unchanged", exc)
        return None
    values = [float(s) for s in scores]
    if len(values) != len(window):
        logger.warning(
            "rank: cross-encoder returned %d score(s) for %d hit(s) — order unchanged",
            len(values), len(window),
        )
        return None
    positions = sorted(range(len(window)), key=lambda i: (-values[i], i))
    return [window[i].row for i in positions]


# ---------------------------------------------------------------------------
# query expansion
# ---------------------------------------------------------------------------

_EXPANSION_SYSTEM = (
    "You rewrite a search query for a personal knowledge vault of university "
    "lecture notes, meeting notes and project notes. Return short paraphrases "
    "that use DIFFERENT vocabulary from the query (synonyms, the formal term "
    "for an informal one and vice versa, the concept a description is naming). "
    "Do not answer the question. Do not add words that narrow the query to a "
    "topic it never mentioned."
)

# The user message, as a template: the number of paraphrases asked for is IN
# the prompt, so two counts are two different questions with two different
# answers, and the key below has to digest it along with the system prompt.
_EXPANSION_USER = "Search query: {query}\n\nReturn {variants} paraphrases."


@dataclass(frozen=True)
class _CachedExpansion:
    """One entry of the on-disk paraphrase cache.

    ``requested`` records how many paraphrases the call that produced this
    entry asked for, so a model that legitimately returned fewer (after
    dropping the query itself and duplicates) is still a cache HIT rather
    than an LLM call re-paid on every search. ``seen`` is a monotonic
    insertion counter: the file is written ``sort_keys=True``, so dict order
    after a reload is alphabetical and cannot be used for eviction.
    """

    variants: tuple[str, ...]
    requested: int
    seen: int


def _cache_path(paths: VaultPaths) -> Path:
    return paths.metadata / _EXPANSION_CACHE_DIR / _EXPANSION_CACHE_FILE


def _expansion_key(query: str, cfg: RankingConfig) -> str:
    """``<digest>:<query>`` — the query, prefixed by a digest of everything
    ELSE that determines the answer.

    Keying on the query alone meant an edited ``_EXPANSION_SYSTEM``, a raised
    ``BRAIN_QUERY_EXPANSION_VARIANTS`` or a switched provider/model kept
    serving the paraphrases produced by the OLD prompt, forever and silently:
    every subsequent A/B re-measured a prompt that no longer existed. The
    digest is computed at call time, not at import, so a test (or an operator
    editing the module) gets the new key immediately. The query stays in the
    key in clear text so the file remains greppable by a human.
    """
    from . import summarize

    # Package-private, not module-private: summarize exposes no public
    # accessor for the resolved provider/model, and re-deriving them from the
    # environment here would silently drift from the router's own rules.
    provider = summarize._select_provider() or ""
    model = summarize._select_model(provider) if provider else ""
    material = "\x00".join(
        [
            _EXPANSION_SYSTEM,
            _EXPANSION_USER,
            str(cfg.query_expansion_variants),
            provider,
            model,
        ]
    )
    return f"{hashlib.sha256(material.encode('utf-8')).hexdigest()[:12]}:{query}"


def _entry_from_json(obj: object) -> _CachedExpansion | None:
    """One decoded cache value, or None when it is not an entry at all.

    ``requested``/``seen`` fall back to 0 rather than rejecting the entry: a
    hand-edited file is still usable, its entries just have to satisfy the
    length check in ``expand_query`` and are evicted first.
    """
    if not isinstance(obj, dict):
        return None
    variants = obj.get("variants")
    if not isinstance(variants, list):
        return None
    requested = obj.get("requested")
    seen = obj.get("seen")
    return _CachedExpansion(
        variants=tuple(v for v in variants if isinstance(v, str)),
        requested=requested if isinstance(requested, int) and not isinstance(requested, bool) else 0,
        seen=seen if isinstance(seen, int) and not isinstance(seen, bool) else 0,
    )


def _read_expansion_cache(paths: VaultPaths) -> dict[str, _CachedExpansion]:
    path = _cache_path(paths)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    out: dict[str, _CachedExpansion] = {}
    for key, value in decoded.items():
        entry = _entry_from_json(value)
        if isinstance(key, str) and entry is not None:
            out[key] = entry
    return out


def _write_expansion_cache(paths: VaultPaths, cache: dict[str, _CachedExpansion]) -> None:
    if len(cache) > _MAX_CACHED_EXPANSIONS:
        # Evict the LEAST RECENTLY WRITTEN, by the ``seen`` counter. Dropping
        # the front of the dict instead would evict whichever key sorts first
        # — the file is written with sort_keys=True, so after a reload the
        # front is the alphabetically-first query, which is as likely to be
        # the newest entry as the oldest. Losing an entry costs one LLM call,
        # nothing else.
        keys = sorted(cache, key=lambda k: (cache[k].seen, k))
        cache = {k: cache[k] for k in keys[len(cache) - _MAX_CACHED_EXPANSIONS:]}
    path = _cache_path(paths)
    payload = {
        key: {"variants": list(e.variants), "requested": e.requested, "seen": e.seen}
        for key, e in cache.items()
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=0, sort_keys=True),
            prefix=".query-expansion-", suffix=".json",
        )
    except OSError:
        return


def expand_query(
    paths: VaultPaths,
    query: str,
    cfg: RankingConfig,
    logger: logging.Logger,
) -> list[str]:
    """Paraphrases of ``query`` (never including the query itself).

    Cached on disk under ``_expansion_key`` — the query plus a digest of the
    prompt, the requested variant count and the provider/model — so a repeated
    eval run is free and deterministic, while a changed prompt or provider
    re-asks instead of serving an answer to a question no longer being posed.
    Returns ``[]`` when no LLM provider is configured, the call fails, or the
    model returns nothing usable — the caller then searches the original query
    alone.
    """
    stripped = query.strip()
    if not stripped:
        return []
    cache = _read_expansion_cache(paths)
    key = _expansion_key(stripped, cfg)
    hit = cache.get(key)
    if hit is not None and (
        hit.requested >= cfg.query_expansion_variants
        or len(hit.variants) >= cfg.query_expansion_variants
    ):
        return list(hit.variants[: cfg.query_expansion_variants])

    from pydantic import BaseModel, Field

    class QueryParaphrases(BaseModel):
        paraphrases: list[str] = Field(
            description="2-3 short reformulations of the search query, "
            "each using different words from the original."
        )

    from . import summarize

    parsed = summarize.generate_structured(
        system=_EXPANSION_SYSTEM,
        user=_EXPANSION_USER.format(
            query=stripped, variants=cfg.query_expansion_variants
        ),
        schema=QueryParaphrases,
        max_tokens=300,
        logger=logger,
    )
    if not isinstance(parsed, QueryParaphrases):
        return []
    seen = {stripped.lower()}
    variants: list[str] = []
    for raw in parsed.paraphrases:
        text = raw.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        variants.append(text)
    variants = variants[: cfg.query_expansion_variants]
    cache[key] = _CachedExpansion(
        variants=tuple(variants),
        requested=cfg.query_expansion_variants,
        seen=max((e.seen for e in cache.values()), default=0) + 1,
    )
    _write_expansion_cache(paths, cache)
    return variants


# ---------------------------------------------------------------------------
# applying the layers to a fused ranking
# ---------------------------------------------------------------------------

def candidates_for_rows(
    paths: VaultPaths, rows: list[int], meta: list[MetaRow]
) -> list[Candidate]:
    """Reduce fused rows to ``rank.Candidate``s, resolving each knowledge
    note's graph node through ``relations.canonical_entity_id`` — the vault's
    single definition of entity identity — never a path-prefix test. Archive
    sources are not entity notes and are not asked about."""
    from .relations import canonical_entity_id

    node_by_source: dict[str, str | None] = {}
    out: list[Candidate] = []
    for i in rows:
        m = meta[i]
        src = m["source_relative_path"]
        if m["origin"] == KNOWLEDGE_EXTRACTOR and src not in node_by_source:
            node_by_source[src] = canonical_entity_id(src, vault_root=paths.root)
        out.append(
            Candidate(
                row=i,
                source=src,
                origin=m["origin"],
                node_id=node_by_source.get(src),
                text=m["text"],
            )
        )
    return out


def apply_layers(
    paths: VaultPaths,
    query: str,
    fused_all: list[tuple[int, float]],
    meta: list[MetaRow],
    *,
    cfg: RankingConfig,
    log: logging.Logger,
    unit: float,
) -> list[tuple[int, float]]:
    """Graph-adjacency boost, salience and cross-encoder rerank, in that order.

    ``unit`` is the top RRF rank's contribution (``1/(k+1)``), owned by
    ``semantic``: boosts are additive on that scale and bounded by their
    weight times it, so they reorder neighbours without swamping the
    fusion. The rerank then permutes the top ``rerank_top_n`` rows and hands
    them the window's OWN scores in descending order — the returned score
    stays an RRF-scale rank score (see ``SearchHit.score``) rather than
    becoming a cross-encoder logit that ``recency.memory_search`` would
    multiply by a recency weight.
    """
    rows = [i for i, _s in fused_all]
    candidates = candidates_for_rows(paths, rows, meta)
    if cfg.graph_boost or cfg.salience:
        graph = load_graph(paths, log)
        boosts: dict[int, float] = {}
        if cfg.graph_boost:
            for row, boost in graph_boosts(
                candidates, graph, unit=unit, weight=cfg.graph_weight
            ).items():
                boosts[row] = boosts.get(row, 0.0) + boost
        if cfg.salience:
            for row, boost in salience_boosts(
                candidates, graph, unit=unit, weight=cfg.salience_weight
            ).items():
                boosts[row] = boosts.get(row, 0.0) + boost
        fused_all = sorted(
            ((i, s + boosts.get(i, 0.0)) for i, s in fused_all),
            key=lambda kv: (-kv[1], kv[0]),
        )
    if cfg.rerank:
        model = load_reranker(cfg.rerank_model, log)
        if model is not None:
            window = fused_all[: cfg.rerank_top_n]
            by_row = {c.row: c for c in candidates}
            reordered = rerank_rows(
                query, [by_row[i] for i, _s in window], model, log
            )
            if reordered is not None:
                scores = [s for _i, s in window]
                fused_all = (
                    list(zip(reordered, scores, strict=True))
                    + fused_all[len(window):]
                )
    return fused_all
