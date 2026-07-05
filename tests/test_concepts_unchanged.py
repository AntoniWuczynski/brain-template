"""Concept-note churn fix: a rebuild over an unchanged vault must not
rewrite (or re-timestamp) a single note, and a topic change must rewrite
only the affected notes — reported in ConceptStats.written_paths so a
later stage can commit exactly those files."""
from __future__ import annotations

import logging
from pathlib import Path

from ingest_lib.concepts import rebuild_concepts
from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.metadata import IndexRecord, append_record

_LOG = logging.getLogger("test")


def _rec(path: str, topics: list[str]) -> IndexRecord:
    return IndexRecord(
        relative_path=path, source_hash="h-" + path, size_bytes=1, extension=".md",
        extractor="text", status="processed", raw_path="archive/raw/" + path,
        processed_path="archive/processed/" + path,
        index_note_path="knowledge/index/" + path, topics=topics,
        summary="A note about " + path,
    )


def _seed(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    for rec in (_rec("a.md", ["Alpha"]), _rec("b.md", ["Beta"])):
        append_record(paths.metadata_index_jsonl, rec)
    return paths


def _concept_bytes(paths: VaultPaths) -> dict[str, str]:
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((paths.knowledge / "concepts").glob("*.md"))
    }


def test_second_rebuild_over_unchanged_vault_writes_nothing(tmp_path: Path):
    paths = _seed(tmp_path)

    first = rebuild_concepts(paths, logger=_LOG)
    assert first.written == 2
    assert first.unchanged == 0
    assert first.written_paths == (
        "knowledge/concepts/alpha.md",
        "knowledge/concepts/beta.md",
    )
    snapshot = _concept_bytes(paths)
    assert set(snapshot) == {"alpha.md", "beta.md"}

    second = rebuild_concepts(paths, logger=_LOG)
    assert second.written == 0
    assert second.unchanged == 2
    assert second.written_paths == ()
    # Byte-for-byte identical — including the `updated:` timestamps, which
    # now mean "content last changed", not "pipeline last ran".
    assert _concept_bytes(paths) == snapshot


def test_topic_change_rewrites_only_affected_notes(tmp_path: Path):
    paths = _seed(tmp_path)
    rebuild_concepts(paths, logger=_LOG)
    before = _concept_bytes(paths)

    # A new source tagged Alpha changes alpha's source list; beta is untouched.
    append_record(paths.metadata_index_jsonl, _rec("c.md", ["Alpha"]))
    stats = rebuild_concepts(paths, logger=_LOG)

    assert stats.written == 1
    assert stats.unchanged == 1
    assert stats.written_paths == ("knowledge/concepts/alpha.md",)

    after = _concept_bytes(paths)
    assert after["beta.md"] == before["beta.md"]
    assert after["alpha.md"] != before["alpha.md"]
    assert "sources_count: 2" in after["alpha.md"]


def test_orphan_removal_reports_removed_paths(tmp_path: Path):
    # The MCP reindex stage commits deletions too, so removed orphans must
    # be reported by vault-relative path, not just counted.
    paths = _seed(tmp_path)
    first = rebuild_concepts(paths, logger=_LOG)
    assert first.removed_paths == ()

    # b.md drops its Beta tag: the beta concept note becomes an orphan.
    append_record(paths.metadata_index_jsonl, _rec("b.md", ["Alpha"]))
    stats = rebuild_concepts(paths, logger=_LOG)

    assert stats.removed == 1
    assert stats.removed_paths == ("knowledge/concepts/beta.md",)
    assert not (paths.knowledge / "concepts" / "beta.md").exists()


def test_user_tail_edit_survives_and_does_not_trigger_rewrite(tmp_path: Path):
    # The user tail below AUTO-GENERATED-END is part of the canonical
    # render, so a tail-only edit leaves the file equal to what the
    # generator would produce: no rewrite (no timestamp churn), and the
    # hand-written text is untouched.
    paths = _seed(tmp_path)
    rebuild_concepts(paths, logger=_LOG)

    alpha = paths.knowledge / "concepts" / "alpha.md"
    edited = alpha.read_text(encoding="utf-8") + "\nMy own thoughts on Alpha.\n"
    alpha.write_text(edited, encoding="utf-8")

    stats = rebuild_concepts(paths, logger=_LOG)
    assert stats.written == 0
    assert stats.unchanged == 2
    assert stats.written_paths == ()
    assert alpha.read_text(encoding="utf-8") == edited
