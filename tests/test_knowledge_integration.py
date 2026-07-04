"""Knowledge notes are first-class enrichment sources: concept notes list
them alongside ingested sources, and their topics feed the connection
graph. (The semantic-index merge is exercised by the live pipeline, per
the repo convention for embeddings glue.)"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ingest_lib.concepts import rebuild_concepts
from ingest_lib.config import paths_for_root
from ingest_lib.connections import rebuild_connections, related_concepts

_LOG = logging.getLogger("test")


def _vault(tmp_path: Path) -> Path:
    for sub in (
        "knowledge/projects", "knowledge/index", "knowledge/concepts",
        "metadata", "inbox", "logs",
        "archive/raw", "archive/processed", "archive/failed",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _ingested_record() -> dict:
    return {
        "relative_path": "uni/lec.txt",
        "source_hash": "a" * 64,
        "size_bytes": 10,
        "extension": ".txt",
        "extractor": "text",
        "status": "processed",
        "raw_path": "archive/raw/uni/lec.txt",
        "processed_path": "archive/processed/uni/lec.md",
        "index_note_path": "knowledge/index/uni/lec.md",
        "assets": [],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "error": None,
        "notes": [],
        "summary": "A lecture about PRNGs.",
        "key_points": [],
        "topics": ["prng"],
    }


PROJECT_NOTE = """---
title: "randeval"
type: project
topics: [prng, randomness]
---

# randeval

## Overview
A randomness evaluation harness.
"""


def _setup(tmp_path: Path):
    root = _vault(tmp_path)
    (root / "metadata/index.jsonl").write_text(
        json.dumps(_ingested_record()) + "\n", encoding="utf-8"
    )
    note = root / "knowledge/projects/randeval/randeval.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(PROJECT_NOTE, encoding="utf-8")
    return paths_for_root(root)


def test_concept_note_lists_knowledge_note_alongside_source(tmp_path: Path) -> None:
    paths = _setup(tmp_path)

    rebuild_concepts(paths, logger=_LOG)

    prng = (paths.knowledge / "concepts/prng.md").read_text(encoding="utf-8")
    assert "[[knowledge/index/uni/lec]]" in prng
    assert "[[knowledge/projects/randeval/randeval]]" in prng
    assert "sources_count: 2" in prng


def test_topic_only_on_knowledge_note_gets_concept_note(tmp_path: Path) -> None:
    paths = _setup(tmp_path)

    rebuild_concepts(paths, logger=_LOG)

    randomness = paths.knowledge / "concepts/randomness.md"
    assert randomness.exists()
    text = randomness.read_text(encoding="utf-8")
    assert "[[knowledge/projects/randeval/randeval]]" in text
    # Snippet comes from the note's first body paragraph.
    assert "randomness evaluation harness" in text


def test_case_variant_topics_count_source_once(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    note = paths.knowledge / "projects/dup/dup.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "---\ntitle: dup\ntopics: [RNG, rng]\n---\n\nBody paragraph.\n",
        encoding="utf-8",
    )

    rebuild_concepts(paths, logger=_LOG)

    rng = (paths.knowledge / "concepts/rng.md").read_text(encoding="utf-8")
    assert "sources_count: 1" in rng
    assert rng.count("[[knowledge/projects/dup/dup]]") == 1


def test_transient_read_error_does_not_remove_orphan_concepts(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = _setup(tmp_path)
    rebuild_concepts(paths, logger=_LOG)
    randomness = paths.knowledge / "concepts/randomness.md"
    assert randomness.exists()  # sourced only from the knowledge note

    # Simulate a transient I/O failure reading the knowledge note: the
    # orphan-removal pass must NOT treat its concepts as orphaned.
    real = Path.read_text

    def flaky(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.name == "randeval.md" and "knowledge/projects" in str(self):
            raise OSError(5, "simulated EIO")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    rebuild_concepts(paths, logger=_LOG)

    assert randomness.exists(), "transient read error must not delete concept notes"


def test_knowledge_topics_feed_connection_graph(tmp_path: Path) -> None:
    paths = _setup(tmp_path)

    rebuild_connections(paths, logger=_LOG)

    # prng<->randomness co-occur only on the knowledge note.
    slug, related = related_concepts(paths, "randomness")
    assert slug == "randomness"
    assert any(r.slug == "prng" and r.cooccurrence >= 1.0 for r in related)
