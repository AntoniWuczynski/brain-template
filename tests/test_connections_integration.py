"""Integration test: connection graph + concept-note linking on a temp vault.

Exercises the file I/O and concept-note rendering end to end (no embeddings,
so only the co-occurrence signal is in play) and asserts determinism.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ingest_lib import rebuild_concepts, rebuild_connections
from ingest_lib.config import VaultPaths
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
        relative_path=path,
        source_hash="h-" + path,
        size_bytes=1,
        extension=".md",
        extractor="text",
        status="processed",
        raw_path="archive/raw/" + path,
        processed_path="archive/processed/" + path,
        index_note_path="knowledge/index/" + path,
        topics=topics,
        summary="A note about " + path,
    )


def test_graph_and_related_links_render_and_are_deterministic(tmp_path: Path):
    paths = _vault(tmp_path)
    paths.ensure()
    for rec in (
        _rec("a.md", ["Alpha", "Beta"]),
        _rec("b.md", ["Beta", "Gamma"]),
        _rec("c.md", ["Alpha", "Beta", "Gamma"]),
    ):
        append_record(paths.metadata_index_jsonl, rec)

    conn = rebuild_connections(paths, logger=_LOG)
    assert conn.cooccurrence_edges == 3        # AB, AG, BG
    assert conn.semantic_edges == 0            # no embeddings present

    connections_path = paths.metadata / "connections.jsonl"
    first_graph = connections_path.read_text(encoding="utf-8")
    assert '"a": "alpha"' in first_graph and '"b": "beta"' in first_graph
    assert '"kind": "cooccurrence"' in first_graph

    rebuild_concepts(paths, logger=_LOG, related=conn.related)
    beta_note = (paths.knowledge / "concepts" / "beta.md").read_text(encoding="utf-8")
    assert "## Related concepts" in beta_note
    # Beta co-occurs with both Alpha and Gamma.
    assert "[[knowledge/concepts/alpha]]" in beta_note
    assert "[[knowledge/concepts/gamma]]" in beta_note
    assert "co-occurs in 2 docs" in beta_note   # Alpha+Beta in a.md and c.md

    # The Related block lives inside the auto-generated zone (split on the
    # real marker comment, not the instruction text that mentions it).
    auto = beta_note.split("<!-- AUTO-GENERATED-END -->")[0]
    assert "## Related concepts" in auto

    # Determinism: a second rebuild reproduces byte-identical artifacts.
    conn2 = rebuild_connections(paths, logger=_LOG)
    assert connections_path.read_text(encoding="utf-8") == first_graph
    rebuild_concepts(paths, logger=_LOG, related=conn2.related)
    assert (paths.knowledge / "concepts" / "beta.md").read_text(encoding="utf-8") == beta_note
