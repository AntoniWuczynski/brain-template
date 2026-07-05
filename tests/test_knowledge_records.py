"""Tests for ingest_lib.knowledge — synthesizing virtual IndexRecords from
curated knowledge/ notes so concepts, semantic search, and the connection
graph treat hand-written notes as first-class sources."""
from __future__ import annotations

from pathlib import Path

from ingest_lib.config import paths_for_root
from ingest_lib.knowledge import knowledge_records


def _vault(tmp_path: Path) -> Path:
    for sub in (
        "knowledge/projects", "knowledge/notes", "knowledge/index",
        "knowledge/concepts", "metadata", "inbox", "logs",
        "archive/raw", "archive/processed", "archive/failed",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


PROJECT_NOTE = """---
title: "randeval"
type: project
status: active
source_repo: "git@github.com:acme/randeval.git"
created: "2026-06-09T09:50:42Z"
updated: "2026-06-09T09:50:42Z"
topics: [randomness, prng, statistical-testing]
aliases: []
---

# randeval

## Overview
A randomness evaluation harness for PRNG and TRNG output streams.

## Stack
Python.
"""


def test_synthesizes_record_from_frontmatter(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/randeval/randeval.md", PROJECT_NOTE)

    records = knowledge_records(paths_for_root(root))

    assert len(records) == 1
    rec = records[0]
    assert rec.relative_path == "knowledge/projects/randeval/randeval.md"
    assert rec.topics == ["randomness", "prng", "statistical-testing"]
    assert rec.status == "processed"
    assert rec.extractor == "knowledge-note"
    assert rec.extension == ".md"
    # The note IS its own readable markdown: semantic search reads
    # processed_path, concept notes wikilink index_note_path.
    assert rec.processed_path == "knowledge/projects/randeval/randeval.md"
    assert rec.index_note_path == "knowledge/projects/randeval/randeval.md"
    assert len(rec.source_hash) == 64
    # Snippet for concept-note source lines: first body paragraph.
    assert "randomness evaluation harness" in rec.summary


def test_skips_generated_dirs_and_non_markdown(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/index/uni/lecture.md", "---\ntopics: [x]\n---\nbody\n")
    _write(root, "knowledge/concepts/prng.md", "---\ntopics: [x]\n---\nbody\n")
    _write(root, "knowledge/projects/.gitkeep", "")
    _write(root, "knowledge/projects/p/img.png", "not markdown")

    assert knowledge_records(paths_for_root(root)) == []


def test_note_without_frontmatter_is_included_with_no_topics(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/notes/scratch.md", "Just a plain hand-written note.\n")

    records = knowledge_records(paths_for_root(root))

    assert len(records) == 1
    assert records[0].topics == []
    assert records[0].relative_path == "knowledge/notes/scratch.md"


def test_malformed_frontmatter_still_included(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/notes/bad.md", "---\ntopics: [unclosed\n---\nbody text\n")

    records = knowledge_records(paths_for_root(root))

    assert len(records) == 1
    assert records[0].topics == []


def test_empty_or_whitespace_notes_skipped(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/notes/empty.md", "")
    _write(root, "knowledge/notes/blank.md", "   \n\n  ")

    assert knowledge_records(paths_for_root(root)) == []


def test_deterministic_ordering(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    for name in ("zeta", "alpha", "mid"):
        _write(root, f"knowledge/notes/{name}.md", f"note {name}\n")

    paths = paths_for_root(root)
    first = [r.relative_path for r in knowledge_records(paths)]
    second = [r.relative_path for r in knowledge_records(paths)]

    assert first == sorted(first)
    assert first == second


def test_non_list_topics_normalised(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/notes/odd.md", "---\ntopics: single-string\n---\nbody\n")

    records = knowledge_records(paths_for_root(root))

    assert len(records) == 1
    assert records[0].topics == ["single-string"]
