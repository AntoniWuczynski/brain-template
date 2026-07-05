"""Tests for querying the persisted concept graph (backs the vault_related tool)."""
from __future__ import annotations

import logging
from pathlib import Path

from ingest_lib import rebuild_connections
from ingest_lib.config import VaultPaths
from ingest_lib.connections import load_edges, related_concepts
from ingest_lib.metadata import IndexRecord, append_record

_LOG = logging.getLogger("test")


def _vault(root: Path) -> VaultPaths:
    return VaultPaths(
        root=root,
        inbox=root / "inbox",
        archive_raw=root / "archive" / "raw",
        archive_processed=root / "archive" / "processed",
        archive_failed=root / "archive" / "failed",
        knowledge=root / "knowledge",
        knowledge_index=root / "knowledge" / "index",
        metadata=root / "metadata",
        metadata_index_jsonl=root / "metadata" / "index.jsonl",
        logs=root / "logs",
    )


def _rec(path: str, topics: list[str]) -> IndexRecord:
    return IndexRecord(
        relative_path=path, source_hash="h-" + path, size_bytes=1, extension=".md",
        extractor="text", status="processed", raw_path="archive/raw/" + path,
        processed_path="archive/processed/" + path, index_note_path=None, topics=topics,
    )


def _seed(paths: VaultPaths) -> None:
    paths.ensure()
    for rec in (
        _rec("a.md", ["Alpha", "Beta"]),
        _rec("b.md", ["Beta", "Gamma"]),
        _rec("c.md", ["Alpha", "Beta", "Gamma"]),
    ):
        append_record(paths.metadata_index_jsonl, rec)
    rebuild_connections(paths, logger=_LOG)  # writes metadata/connections.jsonl


def test_load_edges_reads_persisted_graph(tmp_path: Path):
    paths = _vault(tmp_path)
    _seed(paths)
    edges = load_edges(paths)
    assert edges  # non-empty
    assert all(e.kind in ("cooccurrence", "semantic") for e in edges)
    pairs = {(e.a, e.b) for e in edges if e.kind == "cooccurrence"}
    assert ("alpha", "beta") in pairs


def test_load_edges_absent_graph_is_empty(tmp_path: Path):
    paths = _vault(tmp_path)
    paths.ensure()
    assert load_edges(paths) == []


def test_related_concepts_resolves_slug_or_display(tmp_path: Path):
    paths = _vault(tmp_path)
    _seed(paths)
    # Beta co-occurs with both Alpha and Gamma.
    slug, rels = related_concepts(paths, "Beta", top_n=8)
    assert slug == "beta"
    assert {r.slug for r in rels} == {"alpha", "gamma"}
    # Same answer when queried by slug.
    slug2, rels2 = related_concepts(paths, "beta", top_n=8)
    assert slug2 == "beta"
    assert {r.slug for r in rels2} == {"alpha", "gamma"}


def test_related_concepts_unknown_returns_empty(tmp_path: Path):
    paths = _vault(tmp_path)
    _seed(paths)
    slug, rels = related_concepts(paths, "Nonexistent Topic", top_n=8)
    assert slug == ""
    assert rels == []
