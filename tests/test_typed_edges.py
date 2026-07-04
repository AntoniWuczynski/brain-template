"""Typed entity edges in the connection graph: direction, persistence in
connections.jsonl alongside the concept kinds, and the related_entities
query that backs the MCP entity tools."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.connections import (
    Edge,
    build_related_map,
    compute_connections,
    load_edges,
    rebuild_connections,
    related_entities,
    typed_edges,
)
from ingest_lib.metadata import IndexRecord, append_record
from ingest_lib.relations import EntityInfo, Relation

_LOG = logging.getLogger("test")


def _entity(node_id: str, relations: tuple[Relation, ...], title: str = "") -> EntityInfo:
    return EntityInfo(
        node_id=node_id,
        rel_path=f"knowledge/{node_id}.md",
        title=title or node_id.rsplit("/", 1)[-1],
        type="",
        aliases=(),
        relations=relations,
        updated="",
    )


def _rec(path: str, topics: list[str]) -> IndexRecord:
    return IndexRecord(
        relative_path=path, source_hash="h-" + path, size_bytes=1, extension=".md",
        extractor="text", status="processed", raw_path="archive/raw/" + path,
        processed_path="archive/processed/" + path, index_note_path=None, topics=topics,
    )


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# typed_edges (pure)
# ---------------------------------------------------------------------------

def test_typed_edges_are_directional_not_reordered():
    # "people/zed" > "organisations/acme" lexicographically; an undirected
    # kind would swap them — typed edges must keep origin in `a`.
    entities = {
        "people/zed": _entity(
            "people/zed",
            (Relation(rel="works_at", target="organisations/acme", valid_from="2025-01-01"),),
        )
    }
    edges = typed_edges(entities)
    assert len(edges) == 1
    e = edges[0]
    assert (e.a, e.b) == ("people/zed", "organisations/acme")
    assert e.kind == "typed"
    assert e.rel == "works_at"
    assert e.valid_from == "2025-01-01"
    assert e.valid_until == ""
    assert e.weight == 1.0
    assert e.sources == ("knowledge/people/zed.md",)   # declaring note = provenance


def test_typed_edges_deterministic_sort_and_history_kept():
    entities = {
        "people/zed": _entity(
            "people/zed",
            (
                Relation(rel="works_at", target="organisations/beta", valid_from="2026-01-01"),
                Relation(rel="works_at", target="organisations/acme",
                         valid_from="2024-01-01", valid_until="2025-12-31"),
                Relation(rel="works_at", target="organisations/acme", valid_from="2026-01-01"),
            ),
        ),
        "people/abe": _entity(
            "people/abe",
            (Relation(rel="member_of", target="organisations/acme"),),
        ),
    }
    first = typed_edges(entities)
    second = typed_edges(entities)
    assert first == second
    keys = [(e.a, e.b, e.rel, e.valid_from) for e in first]
    assert keys == sorted(keys)
    # Both acme spans (history) survive as separate edges.
    acme = [e for e in first if e.a == "people/zed" and e.b == "organisations/acme"]
    assert len(acme) == 2


# ---------------------------------------------------------------------------
# compute_connections + jsonl round-trip
# ---------------------------------------------------------------------------

PERSON_NOTE = """---
title: Anna Kowalska
type: person
relations:
  - rel: works_at
    target: organisations/acme
    valid_from: "2025-03-01"
---

# Anna Kowalska
"""


def _seed_three_kinds(tmp_path: Path) -> VaultPaths:
    """A vault where all three signals fire: a.md carries two topics
    (cooccurrence), fabricated embeddings make alpha~beta similar
    (semantic), and a person note declares a relation (typed)."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    for rec in (_rec("a.md", ["Alpha", "Beta"]), _rec("b.md", ["Alpha"]), _rec("c.md", ["Gamma"])):
        append_record(paths.metadata_index_jsonl, rec)

    # Tiny offline embedding index: alpha/beta centroids identical, gamma
    # orthogonal — after mean-centring only alpha~beta clears the floor.
    import numpy as np

    np.save(paths.metadata / "embeddings.npy",
            np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]], dtype=np.float32))
    meta = [{"source_relative_path": p} for p in ("a.md", "b.md", "c.md")]
    (paths.metadata / "embeddings_meta.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in meta), encoding="utf-8"
    )

    _write(paths, "knowledge/people/anna-kowalska.md", PERSON_NOTE)
    return paths


def test_compute_connections_merges_all_three_kinds(tmp_path: Path):
    paths = _seed_three_kinds(tmp_path)
    edges, _related, stats = compute_connections(paths)

    assert {e.kind for e in edges} == {"cooccurrence", "semantic", "typed"}
    assert stats.cooccurrence_edges == 1   # alpha-beta via a.md
    assert stats.semantic_edges == 1       # alpha-beta centroids
    assert stats.typed_edges == 1          # anna -> acme
    typed = [e for e in edges if e.kind == "typed"]
    assert (typed[0].a, typed[0].b) == ("people/anna-kowalska", "organisations/acme")


def test_jsonl_round_trip_preserves_typed_keys_and_old_lines(tmp_path: Path):
    paths = _seed_three_kinds(tmp_path)
    rebuild_connections(paths, logger=_LOG)

    rows = [
        json.loads(ln)
        for ln in (paths.metadata / "connections.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for row in rows:
        if row["kind"] == "typed":
            assert set(row) == {"a", "b", "kind", "weight", "sources",
                                "rel", "valid_from", "valid_until"}
        else:
            # Pre-typed key set, byte-compatible with the old format.
            assert set(row) == {"a", "b", "kind", "weight", "sources"}

    loaded = load_edges(paths)
    typed = [e for e in loaded if e.kind == "typed"]
    assert typed == [
        Edge(a="people/anna-kowalska", b="organisations/acme", kind="typed",
             weight=1.0, sources=("knowledge/people/anna-kowalska.md",),
             rel="works_at", valid_from="2025-03-01", valid_until="")
    ]
    old = [e for e in loaded if e.kind != "typed"]
    assert all(e.rel == "" and e.valid_from == "" and e.valid_until == "" for e in old)


def test_build_related_map_ignores_typed_kind_without_raising():
    edges = [
        Edge(a="alpha", b="beta", kind="cooccurrence", weight=1.0),
        Edge(a="people/x", b="organisations/y", kind="typed", weight=1.0,
             sources=("knowledge/people/x.md",), rel="works_at"),
    ]
    related = build_related_map(edges, {}, top_n=8)   # must not KeyError
    assert "people/x" not in related
    assert [r.slug for r in related["alpha"]] == ["beta"]


# ---------------------------------------------------------------------------
# related_entities
# ---------------------------------------------------------------------------

ANNA = """---
title: Anna Kowalska
type: person
aliases: [Ania]
relations:
  - rel: works_at
    target: organisations/acme
    valid_from: "2023-01-01"
    valid_until: "2024-12-31"
  - rel: works_at
    target: organisations/beta-corp
    valid_from: "2025-01-01"
---

# Anna Kowalska
"""

MEETING = """---
title: Kern call
type: meeting
relations:
  - rel: attended
    target: people/anna-kowalska
---

# Kern call
"""


def _seed_entities(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(paths, "knowledge/people/anna-kowalska.md", ANNA)
    _write(paths, "knowledge/organisations/acme.md",
           "---\ntitle: ACME\ntype: organisation\n---\n\n# ACME\n")
    _write(paths, "knowledge/organisations/beta-corp.md",
           "---\ntitle: Beta Corp\ntype: organisation\n---\n\n# Beta Corp\n")
    _write(paths, "knowledge/meetings/2026/2026-06-12-kern-call.md", MEETING)
    rebuild_connections(paths, logger=_LOG)   # neighbours read PERSISTED state
    return paths


def test_related_entities_resolution_by_id_stem_title_and_alias(tmp_path: Path):
    paths = _seed_entities(tmp_path)
    for query in ("people/anna-kowalska", "anna-kowalska", "Anna Kowalska", "Ania"):
        node, neighbours = related_entities(paths, query)
        assert node == "people/anna-kowalska", query
        assert neighbours, query
    node, neighbours = related_entities(paths, "nobody-here")
    assert (node, neighbours) == ("", [])


def test_related_entities_directions_and_ranking(tmp_path: Path):
    paths = _seed_entities(tmp_path)
    node, neighbours = related_entities(paths, "people/anna-kowalska")
    assert node == "people/anna-kowalska"

    by_id = {n.node_id: n for n in neighbours}
    # Outgoing: relations Anna declared.
    assert by_id["organisations/beta-corp"].direction == "out"
    assert by_id["organisations/beta-corp"].display == "Beta Corp"
    assert by_id["organisations/beta-corp"].source == "knowledge/people/anna-kowalska.md"
    # Incoming: the meeting declared `attended -> anna`.
    meeting = by_id["meetings/2026/2026-06-12-kern-call"]
    assert meeting.direction == "in"
    assert meeting.rel == "attended"
    assert meeting.source == "knowledge/meetings/2026/2026-06-12-kern-call.md"

    # Ranking: current relations first (attended < works_at alphabetically),
    # the ended ACME stint strictly last.
    assert [n.node_id for n in neighbours] == [
        "meetings/2026/2026-06-12-kern-call",
        "organisations/beta-corp",
        "organisations/acme",
    ]
    assert neighbours[-1].valid_until == "2024-12-31"


def test_related_entities_from_target_side_sees_incoming(tmp_path: Path):
    paths = _seed_entities(tmp_path)
    node, neighbours = related_entities(paths, "ACME")   # resolves via title
    assert node == "organisations/acme"
    assert len(neighbours) == 1
    n = neighbours[0]
    assert n.node_id == "people/anna-kowalska"
    assert n.direction == "in"
    assert n.display == "Anna Kowalska"
    assert n.valid_until == "2024-12-31"


def test_related_entities_caps_at_top_n(tmp_path: Path):
    paths = _seed_entities(tmp_path)
    _node, neighbours = related_entities(paths, "people/anna-kowalska", top_n=1)
    assert len(neighbours) == 1
    assert neighbours[0].node_id == "meetings/2026/2026-06-12-kern-call"
