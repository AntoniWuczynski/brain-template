"""Concept-and-entity graph — deterministic, file-native.

Discovers how concepts relate to each other from data already in the vault,
without a graph database. Three signals, all deterministic:

- **co-occurrence**: how often two concepts are tagged on the same document
  (``IndexRecord.topics``). Weight is the document count; sources are the
  documents where both appear.
- **semantic**: cosine similarity between concept *centroids* — the mean of
  the embedding vectors of every chunk belonging to a concept's sources
  (``metadata/embeddings.npy``). Degrades to nothing when no index exists.
- **typed**: explicit ``relations:`` frontmatter on knowledge entity notes
  (people, organisations, projects, meetings) — see ``relations.py``.
  Unlike the other two kinds these edges are DIRECTIONAL and carry a
  relation name plus a validity window.

Edges are written to ``metadata/connections.jsonl`` (append-mostly, atomic),
and a ranked per-concept "Related concepts" view is handed to the concept-note
generator so each note links its neighbours.

The pure functions here (``cooccurrence_edges``, ``semantic_edges``,
``typed_edges``, ``build_related_map``) are numpy-free and deterministic so
they are cheap to unit-test. The embeddings glue and file I/O live below them.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Mapping, Sequence

from .concepts import slugify
from .config import VaultPaths
from .knowledge import knowledge_records
from .metadata import IndexRecord, latest_records_by_path
from .relations import EntityInfo, entity_notes, normalize_target

# Tunables. Conservative defaults — a personal vault, not a recommender system.
_SEMANTIC_TOP_K = 8          # max semantic neighbours kept per concept (kNN)
_SEMANTIC_MIN_COSINE = 0.30  # floor on mean-centred cosine for a semantic edge
_RELATED_TOP_N = 8           # max neighbours surfaced per concept
_RELATED_ENTITIES_TOP_N = 16  # max typed neighbours surfaced per entity

_KIND_COOCCURRENCE = "cooccurrence"
_KIND_SEMANTIC = "semantic"
_KIND_TYPED = "typed"


@dataclass(frozen=True)
class Edge:
    """One relationship between two graph nodes.

    The ``a <= b`` ordering invariant holds for the undirected kinds only
    (cooccurrence/semantic, whose endpoints are concept slugs). Typed edges
    are DIRECTIONAL: ``a`` is the declaring entity, ``b`` the target.
    Entity node ids always contain a ``/`` and concept slugs never do, so
    the two namespaces cannot collide in one file.
    """

    a: str
    b: str
    kind: str                       # _KIND_COOCCURRENCE | _KIND_SEMANTIC | _KIND_TYPED
    weight: float
    sources: tuple[str, ...] = ()   # relative_paths backing the edge
    rel: str = ""                   # typed edges only: relation name
    valid_from: str = ""            # typed edges only: YYYY-MM-DD or ""
    valid_until: str = ""           # typed edges only: "" = currently valid


@dataclass(frozen=True)
class Related:
    """A neighbour of one concept, with the merged signal strengths."""

    slug: str
    display: str
    kinds: tuple[str, ...]
    cooccurrence: float = 0.0
    semantic: float = 0.0


@dataclass(frozen=True)
class TypedNeighbour:
    """A typed-edge neighbour of one entity, as surfaced to query tools."""

    node_id: str
    display: str
    rel: str
    direction: Literal["out", "in"]   # out: entity declared it; in: declared on it
    valid_from: str
    valid_until: str                  # "" = currently valid
    source: str                       # vault-relative path of the declaring note


def _ordered(x: str, y: str) -> tuple[str, str]:
    return (x, y) if x <= y else (y, x)


# ---------------------------------------------------------------------------
# pure signal functions
# ---------------------------------------------------------------------------

def cooccurrence_edges(records: list[IndexRecord]) -> list[Edge]:
    """Edges for every concept pair tagged on the same document.

    Topics are slugified and de-duplicated per document, so case/punctuation
    drift collapses and a concept never pairs with itself.
    """
    counts: dict[tuple[str, str], float] = defaultdict(float)
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rec in records:
        slugs = sorted({slugify(t) for t in (rec.topics or []) if slugify(t)})
        for i in range(len(slugs)):
            for j in range(i + 1, len(slugs)):
                pair = (slugs[i], slugs[j])
                counts[pair] += 1.0
                sources[pair].add(rec.relative_path)
    edges = [
        Edge(
            a=pair[0],
            b=pair[1],
            kind=_KIND_COOCCURRENCE,
            weight=counts[pair],
            sources=tuple(sorted(sources[pair])),
        )
        for pair in counts
    ]
    edges.sort(key=lambda e: (e.a, e.b))
    return edges


def semantic_edges(
    concept_vectors: Mapping[str, Sequence[float]],
    *,
    top_k: int = _SEMANTIC_TOP_K,
    min_cosine: float = _SEMANTIC_MIN_COSINE,
) -> list[Edge]:
    """Symmetrised k-nearest-neighbour graph over concept centroids.

    For each concept, keep its ``top_k`` most-similar neighbours whose cosine
    clears ``min_cosine``; an undirected edge survives if either endpoint
    chose the other. A kNN graph (rather than a global threshold) stays
    meaningful regardless of the absolute cosine scale — important because
    averaged transformer embeddings are anisotropic. Vectors are assumed
    L2-normalised, so cosine is a dot product. O(n²·dim); fine at vault scale.
    """
    slugs = sorted(concept_vectors)
    best: dict[tuple[str, str], float] = {}
    for si in slugs:
        vi = concept_vectors[si]
        sims: list[tuple[float, str]] = []
        for sj in slugs:
            if sj == si:
                continue
            dot = sum(x * y for x, y in zip(vi, concept_vectors[sj], strict=True))
            if dot >= min_cosine:
                sims.append((dot, sj))
        sims.sort(key=lambda t: (-t[0], t[1]))
        for dot, sj in sims[:top_k]:
            pair = _ordered(si, sj)
            best[pair] = round(float(dot), 6)   # symmetric: same cosine either way
    edges = [
        Edge(a=a, b=b, kind=_KIND_SEMANTIC, weight=weight)
        for (a, b), weight in best.items()
    ]
    edges.sort(key=lambda e: (e.a, e.b))
    return edges


def typed_edges(entities: dict[str, EntityInfo]) -> list[Edge]:
    """One directional edge per declared relation: ``a`` is the declaring
    entity's node id, ``b`` the (already normalised) target node id.

    No ``a <= b`` reordering here — direction is the information (Anna
    ``works_at`` ACME, not the reverse). Duplicate (rel, target) pairs with
    different validity windows are history and all survive. ``sources``
    carries the declaring note's vault-relative path for provenance.
    """
    edges = [
        Edge(
            a=entity.node_id,
            b=relation.target,
            kind=_KIND_TYPED,
            weight=1.0,
            sources=(entity.rel_path,),
            rel=relation.rel,
            valid_from=relation.valid_from,
            valid_until=relation.valid_until,
        )
        for entity in entities.values()
        for relation in entity.relations
    ]
    edges.sort(key=lambda e: (e.a, e.b, e.rel, e.valid_from, e.valid_until))
    return edges


@dataclass
class _MergedSlot:
    """Accumulated signal strengths for one undirected concept pair."""

    cooccurrence: float = 0.0
    semantic: float = 0.0
    kinds: set[str] = field(default_factory=set)


def build_related_map(
    edges: list[Edge],
    displays: dict[str, str],
    *,
    top_n: int = _RELATED_TOP_N,
) -> dict[str, list[Related]]:
    """Per-concept ranked neighbour lists, merging both signals.

    Ranking: concepts linked by *both* signals rank above single-signal ones,
    then by combined strength, then alphabetically (stable, deterministic).
    """
    merged: dict[tuple[str, str], _MergedSlot] = {}
    for e in edges:
        # Only the undirected concept kinds merge here. Typed entity edges
        # live in the same file but are directional and not a concept
        # similarity — skip them (and any future kind) instead of letting
        # the accumulation below misattribute a typed weight.
        if e.kind not in (_KIND_COOCCURRENCE, _KIND_SEMANTIC):
            continue
        key = _ordered(e.a, e.b)
        slot = merged.setdefault(key, _MergedSlot())
        if e.kind == _KIND_COOCCURRENCE:
            slot.cooccurrence += e.weight
        else:
            slot.semantic += e.weight
        slot.kinds.add(e.kind)

    neighbours: dict[str, list[Related]] = defaultdict(list)
    for (a, b), slot in merged.items():
        co = slot.cooccurrence
        sem = slot.semantic
        kinds = tuple(sorted(slot.kinds))
        neighbours[a].append(
            Related(slug=b, display=displays.get(b, b), kinds=kinds,
                    cooccurrence=co, semantic=sem)
        )
        neighbours[b].append(
            Related(slug=a, display=displays.get(a, a), kinds=kinds,
                    cooccurrence=co, semantic=sem)
        )

    out: dict[str, list[Related]] = {}
    for slug, rels in neighbours.items():
        rels.sort(key=lambda r: (-len(r.kinds), -(r.cooccurrence + r.semantic), r.slug))
        out[slug] = rels[:top_n]
    return out


# ---------------------------------------------------------------------------
# embeddings glue + orchestration (verified by running the pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConnectionStats:
    concepts: int
    cooccurrence_edges: int
    semantic_edges: int
    related: dict[str, list[Related]]
    typed_edges: int = 0


def _concept_sources(records: list[IndexRecord]) -> dict[str, set[str]]:
    """slug -> set of source ``relative_path``s carrying that topic."""
    out: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        for topic in rec.topics or []:
            slug = slugify(topic)
            if slug:
                out[slug].add(rec.relative_path)
    return out


def _concept_displays(records: list[IndexRecord]) -> dict[str, str]:
    """slug -> most common display variant (matches concepts.py voting)."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        for topic in rec.topics or []:
            slug = slugify(topic)
            if slug:
                votes[slug][topic] += 1
    return {slug: counter.most_common(1)[0][0] for slug, counter in votes.items()}


def concept_vectors_from_embeddings(
    paths: VaultPaths, concept_sources: dict[str, set[str]]
) -> dict[str, list[float]]:
    """Per-concept centroid (mean of member chunks, re-normalised).

    Returns ``{}`` when the index is missing, unreadable, or numpy is absent —
    the semantic signal simply drops out and co-occurrence carries the graph.
    """
    vectors_path = paths.metadata / "embeddings.npy"
    meta_path = paths.metadata / "embeddings_meta.jsonl"
    if not vectors_path.exists() or not meta_path.exists():
        return {}
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return {}

    # A truncated/torn index (crash mid-write, disk error, stray hand-edit)
    # must degrade to the co-occurrence-only graph the docstring promises,
    # not crash 'ingest.py --rebuild-concepts' with a np.load/JSON traceback.
    try:
        vectors = np.load(vectors_path)
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = [json.loads(ln) for ln in fh if ln.strip()]
    except (OSError, ValueError, EOFError):
        return {}
    if len(meta) != vectors.shape[0]:
        return {}

    rows_by_source: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(meta):
        rows_by_source[row["source_relative_path"]].append(i)

    # Raw centroid per concept. Row indices are sorted so the float32
    # summation order is fixed run-to-run — set iteration order is not
    # stable across processes and would otherwise break determinism.
    raw: dict[str, np.ndarray] = {}
    for slug, sources in concept_sources.items():
        idx = sorted(i for src in sources for i in rows_by_source.get(src, []))
        if not idx:
            continue
        raw[slug] = vectors[idx].mean(axis=0)
    if not raw:
        return {}

    # Mean-centre the centroids before normalising. Averaged transformer
    # embeddings are anisotropic (every centroid sits in a narrow cone, so
    # raw pairwise cosine is uniformly high); subtracting the global mean
    # restores discriminative power. Sorted stacking keeps the mean stable.
    ordered = sorted(raw)
    matrix = np.stack([raw[s] for s in ordered])
    global_mean = matrix.mean(axis=0)

    out: dict[str, list[float]] = {}
    for slug in ordered:
        centred = raw[slug] - global_mean
        norm = float(np.linalg.norm(centred))
        if norm == 0.0:
            continue
        out[slug] = (centred / norm).astype(float).tolist()
    return out


def compute_connections(
    paths: VaultPaths,
) -> tuple[list[Edge], dict[str, list[Related]], ConnectionStats]:
    """Build the full edge list + related map from metadata and embeddings."""
    # Knowledge notes count: their topics co-occur and their chunks (once
    # embedded) contribute to concept centroids via source_relative_path.
    records = (
        list(latest_records_by_path(paths.metadata_index_jsonl).values())
        + knowledge_records(paths)
    )
    co_edges = cooccurrence_edges(records)
    concept_sources = _concept_sources(records)
    displays = _concept_displays(records)
    vectors = concept_vectors_from_embeddings(paths, concept_sources)
    sem_edges = semantic_edges(vectors) if vectors else []
    # Entity relations ride along in the same edge list/file: one graph
    # artefact, three signals. build_related_map ignores the typed kind.
    t_edges = typed_edges(entity_notes(paths))
    edges = co_edges + sem_edges + t_edges
    related = build_related_map(edges, displays)
    stats = ConnectionStats(
        concepts=len(concept_sources),
        cooccurrence_edges=len(co_edges),
        semantic_edges=len(sem_edges),
        related=related,
        typed_edges=len(t_edges),
    )
    return edges, related, stats


def rebuild_connections(paths: VaultPaths, *, logger: logging.Logger) -> ConnectionStats:
    """Recompute the concept graph and persist it to ``connections.jsonl``."""
    paths.ensure()
    edges, _related, stats = compute_connections(paths)
    _write_connections_jsonl(paths, edges)
    logger.info(
        "connections: %d concepts, %d co-occurrence + %d semantic + %d typed edge(s)",
        stats.concepts,
        stats.cooccurrence_edges,
        stats.semantic_edges,
        stats.typed_edges,
    )
    return stats


def load_edges(paths: VaultPaths) -> list[Edge]:
    """Read the persisted edge list from ``metadata/connections.jsonl``.

    Returns ``[]`` if the graph hasn't been built yet. Malformed lines are
    skipped rather than failing the whole read.
    """
    path = paths.metadata / "connections.jsonl"
    if not path.exists():
        return []
    edges: list[Edge] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not all(k in row for k in ("a", "b", "kind", "weight")):
            continue
        try:
            edge = Edge(
                a=row["a"],
                b=row["b"],
                kind=row["kind"],
                weight=float(row["weight"]),
                sources=tuple(row.get("sources") or []),
                # Typed-edge keys are absent on cooccurrence/semantic lines.
                rel=str(row.get("rel") or ""),
                valid_from=str(row.get("valid_from") or ""),
                valid_until=str(row.get("valid_until") or ""),
            )
        except (ValueError, TypeError):
            # Structurally present but malformed (non-numeric weight,
            # non-iterable sources): skip it, don't fail the whole read —
            # the docstring promises malformed lines are skipped.
            continue
        edges.append(edge)
    return edges


def related_concepts(
    paths: VaultPaths, query: str, *, top_n: int = _RELATED_TOP_N
) -> tuple[str, list[Related]]:
    """Resolve ``query`` (a concept slug or display name) and return its ranked
    related concepts from the persisted graph.

    Returns ``(resolved_slug, neighbours)``. ``("", [])`` when ``query`` is not
    a known concept; ``(slug, [])`` when it is known but has no edges.
    """
    records = (
        list(latest_records_by_path(paths.metadata_index_jsonl).values())
        + knowledge_records(paths)
    )
    displays = _concept_displays(records)
    slug = slugify(query)
    if slug not in displays:
        return "", []
    related = build_related_map(load_edges(paths), displays, top_n=top_n)
    return slug, related.get(slug, [])


def _resolve_entity(entities: dict[str, EntityInfo], query: str) -> str:
    """Resolve a query string to an entity node id, or "" if unknown.

    Match order (first hit wins, ties broken by sorted node id so the
    answer is deterministic): exact node id, exact path stem, slugified
    title, slugified alias. The node-id check goes through
    ``normalize_target`` so wikilinked/prefixed forms resolve too.
    """
    q = query.strip()
    if not q:
        return ""
    node = normalize_target(q)
    if node in entities:
        return node
    ordered = sorted(entities)
    for nid in ordered:
        if nid.rsplit("/", 1)[-1] == q:
            return nid
    q_slug = slugify(q)
    if q_slug:
        for nid in ordered:
            if slugify(entities[nid].title) == q_slug:
                return nid
        for nid in ordered:
            if any(slugify(a) == q_slug for a in entities[nid].aliases):
                return nid
    return ""


def related_entities(
    paths: VaultPaths, query: str, *, top_n: int = _RELATED_ENTITIES_TOP_N
) -> tuple[str, list[TypedNeighbour]]:
    """Resolve ``query`` to an entity and return its typed neighbours from
    the PERSISTED graph (``connections.jsonl``, via :func:`load_edges` —
    consistent with ``related_concepts`` reading persisted state).

    Both directions are surfaced: relations the entity declared (``out``)
    and relations declared on it by other notes (``in``). Ranking: current
    relations (no valid_until) before ended ones, then rel alphabetically,
    then node id. Returns ``("", [])`` when ``query`` matches no entity.
    """
    entities = entity_notes(paths)
    node = _resolve_entity(entities, query)
    if not node:
        return "", []

    neighbours: list[TypedNeighbour] = []
    for e in load_edges(paths):
        if e.kind != _KIND_TYPED:
            continue
        direction: Literal["out", "in"]
        if e.a == node:
            other, direction = e.b, "out"
        elif e.b == node:
            other, direction = e.a, "in"
        else:
            continue
        info = entities.get(other)
        neighbours.append(
            TypedNeighbour(
                node_id=other,
                display=info.title if info else other,   # dangling target: show the id
                rel=e.rel,
                direction=direction,
                valid_from=e.valid_from,
                valid_until=e.valid_until,
                source=e.sources[0] if e.sources else "",
            )
        )
    neighbours.sort(
        key=lambda n: (bool(n.valid_until), n.rel, n.node_id, n.direction, n.valid_from)
    )
    return node, neighbours[:top_n]


def _write_connections_jsonl(paths: VaultPaths, edges: list[Edge]) -> None:
    """Atomically write the edge list. Deterministic ordering, no timestamps."""
    target = paths.metadata / "connections.jsonl"
    # rel/valid_from tie-break duplicate typed (a, b) pairs (relation
    # history); both are "" on the undirected kinds, so their order — and
    # their serialised lines — stay byte-identical to the pre-typed format.
    ordered = sorted(edges, key=lambda e: (e.a, e.b, e.kind, e.rel, e.valid_from, e.valid_until))
    fd, tmp = tempfile.mkstemp(prefix=".connections-", suffix=".jsonl", dir=str(paths.metadata))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for e in ordered:
                row: dict[str, object] = {
                    "a": e.a,
                    "b": e.b,
                    "kind": e.kind,
                    "weight": e.weight,
                    "sources": list(e.sources),
                }
                if e.kind == _KIND_TYPED:
                    # Only typed lines carry these keys, so existing
                    # cooccurrence/semantic lines remain byte-identical.
                    row["rel"] = e.rel
                    row["valid_from"] = e.valid_from
                    row["valid_until"] = e.valid_until
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
