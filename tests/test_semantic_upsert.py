"""Tests for ingest_lib.semantic.upsert_notes — patching individual
knowledge notes into the embedding index without a full rebuild.

The real embedding model is never loaded: tests seed the index files by
hand with a deterministic hash-based encoder and pass the same encoder
into upsert_notes. Fallback paths monkeypatch build_index to a sentinel.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from ingest_lib import semantic
from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.semantic import upsert_notes

_LOG = logging.getLogger("test")
_DIM = 8


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic per-text unit vectors derived from sha256 — stable
    across calls so vector rows can be matched back to their meta text."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:_DIM], dtype=np.uint8).astype(np.float32) + 1.0
        rows.append(vec / np.linalg.norm(vec))
    return np.vstack(rows)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _seed_index(paths: VaultPaths, rows: list[tuple[str, int, str]]) -> None:
    """Hand-build the two index files from (path, chunk_idx, text) rows so
    build_index (and thus the model) is never invoked."""
    vecs = _fake_encode([text for _, _, text in rows])
    np.save(paths.metadata / "embeddings.npy", vecs)
    with (paths.metadata / "embeddings_meta.jsonl").open("w", encoding="utf-8") as fh:
        for rel, idx, text in rows:
            fh.write(
                json.dumps(
                    {
                        "source_relative_path": rel,
                        "source_hash": "h-" + rel,
                        "title": Path(rel).stem,
                        "chunk_idx": idx,
                        "text": text,
                        "origin": "knowledge-note" if rel.startswith("knowledge/") else "text",
                        "model": "BAAI/bge-small-en-v1.5",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _load_index(paths: VaultPaths) -> tuple[np.ndarray, list[dict]]:
    vecs = np.load(paths.metadata / "embeddings.npy")
    with (paths.metadata / "embeddings_meta.jsonl").open("r", encoding="utf-8") as fh:
        meta = [json.loads(ln) for ln in fh if ln.strip()]
    return vecs, meta


# Long enough (> _MIN_CHARS = 80) to survive chunking.
_KEPT_TEXT = (
    "An unrelated archived source paragraph that must survive every upsert "
    "completely untouched, byte for byte."
)

_NOTE_V1 = """---
title: "anna kowalska"
type: person
updated: "2026-06-10T10:00:00Z"
topics: [people]
---

# Anna Kowalska

Anna is a research engineer at Acme who collaborates on the kern project and
reviews the ingestion pipeline design documents with the team every quarter.
"""

# Two ~900-char paragraphs: greedy packing (~1600-char budget) puts them
# in separate chunks, so this note contributes exactly two rows.
_PARA_A = ("alpha " * 150).strip()
_PARA_B = ("bravo " * 150).strip()
_NOTE_TWO_CHUNKS = f"---\ntitle: big\nupdated: \"2026-06-01\"\n---\n\n{_PARA_A}\n\n{_PARA_B}\n"


def test_upsert_appends_rows_and_keeps_alignment(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    _write(paths, "knowledge/people/anna-kowalska.md", _NOTE_V1)

    n = upsert_notes(
        paths, ["knowledge/people/anna-kowalska.md"], logger=_LOG, encode=_fake_encode
    )

    assert n == 1
    vecs, meta = _load_index(paths)
    # Searchable state stays consistent: every vector row pairs with its
    # meta row, and each row's vector is the encoding of its own text.
    assert vecs.shape[0] == len(meta) == 2
    for row, m in zip(vecs, meta, strict=True):
        assert np.allclose(row, _fake_encode([m["text"]])[0])
    # Untouched source kept verbatim, new rows appended after it.
    assert meta[0]["source_relative_path"] == "uni/lecture.md"
    assert meta[0]["text"] == _KEPT_TEXT
    assert meta[1]["source_relative_path"] == "knowledge/people/anna-kowalska.md"
    assert meta[1]["chunk_idx"] == 0
    assert meta[1]["origin"] == "knowledge-note"
    assert meta[1]["title"] == "anna kowalska"  # stem with dashes -> spaces
    assert "research engineer at Acme" in meta[1]["text"]


def test_reupsert_replaces_rows_including_count_changes(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    rel = "knowledge/notes/big.md"
    _write(paths, rel, _NOTE_TWO_CHUNKS)

    assert upsert_notes(paths, [rel], logger=_LOG, encode=_fake_encode) == 2
    vecs, meta = _load_index(paths)
    assert vecs.shape[0] == len(meta) == 3
    assert [m["chunk_idx"] for m in meta if m["source_relative_path"] == rel] == [0, 1]

    # Shrink the note to one chunk: old rows replaced, count drops.
    _write(paths, rel, "---\ntitle: big\n---\n\n" + _KEPT_TEXT + " But now rewritten.\n")
    assert upsert_notes(paths, [rel], logger=_LOG, encode=_fake_encode) == 1
    vecs, meta = _load_index(paths)
    assert vecs.shape[0] == len(meta) == 2
    note_rows = [m for m in meta if m["source_relative_path"] == rel]
    assert len(note_rows) == 1
    assert "But now rewritten" in note_rows[0]["text"]
    assert _PARA_A not in json.dumps(meta)  # stale chunk really gone
    # The unrelated source survived both upserts.
    assert meta[0]["source_relative_path"] == "uni/lecture.md"


def test_deleted_note_rows_are_dropped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    gone = "knowledge/notes/gone.md"
    _seed_index(
        paths,
        [("uni/lecture.md", 0, _KEPT_TEXT), (gone, 0, "stale text for a deleted note " * 4)],
    )
    # No file exists at `gone` — the upsert must treat that as deletion.
    n = upsert_notes(paths, [gone], logger=_LOG, encode=_fake_encode)

    assert n == 0
    vecs, meta = _load_index(paths)
    assert vecs.shape[0] == len(meta) == 1
    assert meta[0]["source_relative_path"] == "uni/lecture.md"


def test_missing_index_falls_back_to_full_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/notes/a.md", _NOTE_V1)
    calls: list[VaultPaths] = []

    def fake_build(p: VaultPaths, *, logger: logging.Logger) -> int:
        calls.append(p)
        return 7

    monkeypatch.setattr(semantic, "build_index", fake_build)
    n = upsert_notes(paths, ["knowledge/notes/a.md"], logger=_LOG, encode=_fake_encode)

    assert n == 7
    assert calls == [paths]


def test_row_count_mismatch_falls_back_to_full_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    # Two vectors, one meta row: a torn pair from an interrupted writer.
    np.save(paths.metadata / "embeddings.npy", _fake_encode(["one", "two"]))
    (paths.metadata / "embeddings_meta.jsonl").write_text(
        json.dumps({"source_relative_path": "x.md", "chunk_idx": 0, "text": "one"}) + "\n",
        encoding="utf-8",
    )
    _write(paths, "knowledge/notes/a.md", _NOTE_V1)
    calls: list[VaultPaths] = []

    def fake_build(p: VaultPaths, *, logger: logging.Logger) -> int:
        calls.append(p)
        return 99

    monkeypatch.setattr(semantic, "build_index", fake_build)
    n = upsert_notes(paths, ["knowledge/notes/a.md"], logger=_LOG, encode=_fake_encode)

    assert n == 99
    assert calls == [paths]


def test_non_knowledge_paths_are_skipped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    before = (paths.metadata / "embeddings_meta.jsonl").read_bytes()

    n = upsert_notes(
        paths,
        [
            "archive/processed/uni/lecture.md",  # ingested output, not a note
            "knowledge/index/uni/lecture.md",    # generated, excluded
            "knowledge/concepts/prng.md",        # generated, excluded
            "inbox/scratch.md",                  # not knowledge at all
        ],
        logger=_LOG,
        encode=_fake_encode,
    )

    assert n == 0
    # Nothing valid remained, so the index files were not rewritten.
    assert (paths.metadata / "embeddings_meta.jsonl").read_bytes() == before


def test_empty_note_rows_dropped_and_meetings_dir_allowed(tmp_path: Path) -> None:
    # meetings/ is part of the upsert allowlist (new memory area) even
    # though it ships empty; an empty note's rows are dropped not kept.
    paths = _vault(tmp_path)
    rel = "knowledge/meetings/2026/2026-06-12-kern-call.md"
    _seed_index(paths, [(rel, 0, "stale meeting row text that should disappear " * 3)])
    _write(paths, rel, "   \n\n  ")

    n = upsert_notes(paths, [rel], logger=_LOG, encode=_fake_encode)

    assert n == 0
    vecs, meta = _load_index(paths)
    assert vecs.shape[0] == len(meta) == 0


def test_upsert_skips_assistant_archive_path(tmp_path: Path) -> None:
    # F16/F8: knowledge/assistant/archive/ holds promoted facts already
    # carried by their entity note — upserting one must be a no-op (the
    # consolidate CLI may hand us such destinations), mirroring the scan
    # exclusion. Even though the file exists, it adds/drops nothing.
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    before = (paths.metadata / "embeddings_meta.jsonl").read_bytes()
    archived = "knowledge/assistant/archive/2026/fact-001.md"
    _write(paths, archived, _NOTE_V1)

    n = upsert_notes(paths, [archived], logger=_LOG, encode=_fake_encode)

    assert n == 0
    # Index files untouched: the archive path was skipped, not indexed.
    assert (paths.metadata / "embeddings_meta.jsonl").read_bytes() == before
