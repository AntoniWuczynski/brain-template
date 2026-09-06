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
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from ingest_lib import semantic
from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.knowledge import _record_for_note
from ingest_lib.lexical import LexicalIndex
from ingest_lib.metadata import IndexRecord, Status, append_record
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
    build_index (and thus the model) is never invoked. Written through
    ``_write_index_files`` so the seeded pair carries the same generation
    stamp any index this module writes does."""
    semantic._write_index_files(
        _fake_encode([text for _, _, text in rows]),
        [
            semantic.MetaRow(
                source_relative_path=rel,
                source_hash="h-" + rel,
                title=Path(rel).stem,
                chunk_idx=idx,
                text=text,
                origin="knowledge-note" if rel.startswith("knowledge/") else "text",
                heading_path="",
                model="BAAI/bge-small-en-v1.5",
                gen="",  # stamped by _write_index_files
            )
            for rel, idx, text in rows
        ],
        vectors_path=paths.metadata / "embeddings.npy",
        meta_path=paths.metadata / "embeddings_meta.jsonl",
    )


class _FakeEmbedder:
    """Stands in for the sentence-transformers model on the search path.
    Same deterministic encoder the index is seeded with, so a query that is
    a chunk's exact text scores 1.0 against that chunk's row."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        return _fake_encode(sentences)


def _prepare_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``semantic.search`` usable without the real model: the fake
    encoder, no BGE query instruction (so a query equal to a chunk's text
    encodes to that chunk's vector) and empty process-wide index caches
    (monkeypatch restores them afterwards)."""
    monkeypatch.setenv("BRAIN_QUERY_INSTRUCTION", "0")
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeEmbedder(), "cpu"))
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
    monkeypatch.setattr(semantic, "_LEXICAL_CACHE", None)


class _MetaRow(TypedDict):
    source_relative_path: str
    source_hash: str
    title: str
    chunk_idx: int
    text: str
    origin: str
    model: str


def _load_index(paths: VaultPaths) -> tuple[np.ndarray, list[_MetaRow]]:
    vecs = np.load(paths.metadata / "embeddings.npy")
    with (paths.metadata / "embeddings_meta.jsonl").open("r", encoding="utf-8") as fh:
        meta: list[_MetaRow] = [json.loads(ln) for ln in fh if ln.strip()]
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
_PARA_C = ("gamma " * 150).strip()


def _one_para_note(para: str) -> str:
    """A note whose body is exactly one chunk, equal to ``para``."""
    return f"---\ntitle: n\n---\n\n{para}\n"


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


def test_search_never_pairs_vectors_and_meta_from_two_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search landing between the upsert's two ``os.replace`` calls sees
    the new vectors against the old meta. Rewriting a note keeps the row
    count identical (2 before, 2 after) while moving every row, so a
    row-count guard sees nothing wrong — before the generation stamp this
    returned the query's own vector under the OTHER note's path and
    snippet."""
    paths = _vault(tmp_path)
    a, b = "knowledge/notes/a.md", "knowledge/notes/b.md"
    _seed_index(paths, [(a, 0, _PARA_A), (b, 0, _PARA_B)])
    _write(paths, a, _one_para_note(_PARA_C))   # rewritten: same row count
    _write(paths, b, _one_para_note(_PARA_B))
    _prepare_search(monkeypatch)

    vectors_renamed = threading.Event()
    reader_done = threading.Event()
    real_replace = os.replace

    def paused_replace(src: str | Path, dst: str | Path) -> None:
        real_replace(src, dst)
        if str(dst).endswith("embeddings.npy"):
            vectors_renamed.set()
            reader_done.wait(10)

    monkeypatch.setattr(os, "replace", paused_replace)
    written: list[int] = []
    writer = threading.Thread(
        target=lambda: written.append(
            upsert_notes(paths, [a], logger=_LOG, encode=_fake_encode)
        )
    )
    writer.start()
    try:
        assert vectors_renamed.wait(10)
        hits = semantic.search(paths, _PARA_B, top_k=1, mode="dense", logger=_LOG)
    finally:
        reader_done.set()
        writer.join(10)

    assert written == [1]
    # The old pair or the new pair, never a mix: no hit may carry a's path.
    assert [h.source_relative_path for h in hits] in ([], [b])
    # And the settled index answers the same query correctly.
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
    after = semantic.search(paths, _PARA_B, top_k=1, mode="dense", logger=_LOG)
    assert [h.source_relative_path for h in after] == [b]
    assert after[0].snippet == _PARA_B


def test_torn_index_is_refused_by_search_and_rebuilt_by_upsert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the two renames leaves new vectors beside old meta.
    The row counts agree, so the loader must reject the pair on its
    generation stamp instead of serving rows from the wrong source, and the
    next upsert must rebuild rather than patch rows into it."""
    paths = _vault(tmp_path)
    a, b = "knowledge/notes/a.md", "knowledge/notes/b.md"
    _seed_index(paths, [(a, 0, _PARA_A), (b, 0, _PARA_B)])
    _write(paths, a, _one_para_note(_PARA_C))
    _write(paths, b, _one_para_note(_PARA_B))
    _prepare_search(monkeypatch)

    real_replace = os.replace

    def crashing_replace(src: str | Path, dst: str | Path) -> None:
        real_replace(src, dst)
        if str(dst).endswith("embeddings.npy"):
            raise RuntimeError("power loss between the two renames")

    monkeypatch.setattr(os, "replace", crashing_replace)
    # Pinned to the message: a bare `raises(RuntimeError)` would also pass on
    # an unrelated RuntimeError from inside upsert_notes (AUD-073).
    with pytest.raises(RuntimeError, match="power loss between the two renames"):
        upsert_notes(paths, [a], logger=_LOG, encode=_fake_encode)
    monkeypatch.setattr(os, "replace", real_replace)

    vecs, meta = _load_index(paths)
    assert vecs.shape[0] == len(meta) == 2      # the count guard sees nothing
    assert meta[0]["source_relative_path"] == a  # ...while the rows have moved

    assert semantic.search(paths, _PARA_B, top_k=1, mode="dense", logger=_LOG) == []

    calls: list[VaultPaths] = []

    def fake_build(p: VaultPaths, *, logger: logging.Logger) -> int:
        calls.append(p)
        return 42

    monkeypatch.setattr(semantic, "build_index", fake_build)
    assert upsert_notes(paths, [b], logger=_LOG, encode=_fake_encode) == 42
    assert calls == [paths]


def test_row_hash_matches_the_text_it_was_chunked_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AUD-052: the note used to be read once for its hash and again for its
    # chunks, so an editor writing between the two produced a row whose
    # source_hash named content the row's text did not hold — sweep saw no
    # drift and search served the stale snippet.
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    rel = "knowledge/notes/raced.md"
    v1 = _one_para_note(_PARA_A)
    v2 = _one_para_note(_PARA_C)
    _write(paths, rel, v1)

    def _edit_after_the_record_is_built(md: Path, p: VaultPaths) -> IndexRecord | None:
        rec = _record_for_note(md, p)
        _write(paths, rel, v2)      # a concurrent Obsidian/consolidate write
        return rec

    monkeypatch.setattr(semantic, "_record_for_note", _edit_after_the_record_is_built)
    assert upsert_notes(paths, [rel], logger=_LOG, encode=_fake_encode) == 1

    _vecs, meta = _load_index(paths)
    row = next(m for m in meta if m["source_relative_path"] == rel)
    body, version = (_PARA_A, v1) if _PARA_A in row["text"] else (_PARA_C, v2)
    assert body in row["text"]
    assert row["source_hash"] == hashlib.sha256(version.encode("utf-8")).hexdigest()


def test_upsert_fires_the_sqlite_vec_tripwire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # AUD-097: the tripwire only ran on full rebuilds, but the MCP upsert is
    # the path that grows the index between them.
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    rel = "knowledge/notes/grow.md"
    _write(paths, rel, _one_para_note(_PARA_A))
    monkeypatch.setattr(semantic, "_SQLITE_VEC_HINT_THRESHOLD", 2)

    with caplog.at_level(logging.WARNING, logger="test"):
        assert upsert_notes(paths, [rel], logger=_LOG, encode=_fake_encode) == 1
    assert any("sqlite-vec" in r.getMessage() for r in caplog.records)


# Takes metadata/.embeddings.lock in a separate process and holds it until a
# line arrives on stdin. Deliberately imports nothing from the project: the
# lock is a plain advisory flock on a path, and that is the whole contract.
_LOCK_HOLDER = (
    "import fcntl, sys\n"
    "fh = open(sys.argv[1], 'a')\n"
    "fcntl.flock(fh.fileno(), fcntl.LOCK_EX)\n"
    "sys.stdout.write('locked\\n'); sys.stdout.flush()\n"
    "sys.stdin.readline()\n"
    "fcntl.flock(fh.fileno(), fcntl.LOCK_UN)\n"
)


def test_upsert_waits_for_the_cross_process_index_lock(tmp_path: Path) -> None:
    # AUD-053: metadata/.embeddings.lock is the only thing keeping a CLI
    # rebuild from interleaving with an MCP upsert, and nothing exercised it.
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT_TEXT)])
    rel = "knowledge/notes/contended.md"
    _write(paths, rel, _one_para_note(_PARA_A))

    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, str(paths.metadata / ".embeddings.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert holder.stdin is not None and holder.stdout is not None
    try:
        assert holder.stdout.readline().strip() == "locked"

        done = threading.Event()
        written: list[int] = []

        def _run() -> None:
            written.append(upsert_notes(paths, [rel], logger=_LOG, encode=_fake_encode))
            done.set()

        worker = threading.Thread(target=_run)
        worker.start()
        try:
            # Another process holds the writer lock: the upsert must block,
            # and must not have published anything.
            assert not done.wait(0.5)
            assert len(_load_index(paths)[1]) == 1

            holder.stdin.write("go\n")
            holder.stdin.flush()
            assert done.wait(30)
        finally:
            worker.join(timeout=30)
        assert written == [1]
        assert len(_load_index(paths)[1]) == 2
    finally:
        holder.kill()
        holder.wait(timeout=30)


# ---------------------------------------------------------------------------
# AUD-114: `partial` is not one thing
# ---------------------------------------------------------------------------


def _source_record(
    rel: str, *, status: Status, extractor: str, extension: str, processed: str
) -> IndexRecord:
    return IndexRecord(
        relative_path=rel,
        source_hash="h-" + rel,
        size_bytes=1,
        extension=extension,
        extractor=extractor,
        status=status,
        raw_path=rel,
        processed_path=processed,
        index_note_path=None,
    )


_PPTX_BODY = (
    "The trilobite die-off at the Permian boundary is the single largest "
    "extinction recorded in the marine fossil record, and it is what this "
    "lecture spends its middle third on.\n"
)
_PDF_BODY = (
    "Only the first page of this scanned report came out of the text layer, "
    "which is why the extractor refused to call the job finished at all.\n"
)


def test_build_index_includes_visuals_only_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A .pptx whose TEXT extracted cleanly but which holds an unextracted
    picture is `partial`, and used to be dropped from search entirely; a
    .pdf that fell back to pypdf is `partial` because its text is short of
    the source, and must stay out."""
    paths = _vault(tmp_path)
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeEmbedder(), "cpu"))
    _write(paths, "archive/processed/deck.md", _PPTX_BODY)
    _write(paths, "archive/processed/scan.md", _PDF_BODY)
    for rec in (
        _source_record(
            "archive/raw/deck.pptx", status="partial", extractor="pptx",
            extension=".pptx", processed="archive/processed/deck.md",
        ),
        _source_record(
            "archive/raw/scan.pdf", status="partial", extractor="pdf-pypdf",
            extension=".pdf", processed="archive/processed/scan.md",
        ),
    ):
        append_record(paths.metadata_index_jsonl, rec)

    with caplog.at_level(logging.WARNING, logger="test"):
        written = semantic.build_index(paths, logger=_LOG)

    assert written == 1
    sources = {row["source_relative_path"] for row in _load_index(paths)[1]}
    assert sources == {"archive/raw/deck.pptx"}
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "1 'partial' record(s) are NOT indexed" in warning
    assert ".pdf=1" in warning
    assert ".pptx" not in warning


def test_partial_withheld_warning_drops_mineru_advice_when_no_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MinerU fixes PDFs and nothing else — a withheld .vtt must not be told
    to install it."""
    paths = _vault(tmp_path)
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeEmbedder(), "cpu"))
    _write(paths, "archive/processed/talk.md", _PDF_BODY)
    append_record(
        paths.metadata_index_jsonl,
        _source_record(
            "archive/raw/talk.vtt", status="partial", extractor="audio",
            extension=".vtt", processed="archive/processed/talk.md",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="test"):
        semantic.build_index(paths, logger=_LOG)

    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert ".vtt=1" in warning
    assert "MinerU" not in warning
    assert "Re-extract and re-ingest" in warning


# ---------------------------------------------------------------------------
# AUD-051: the BM25 index is not rebuilt on the search path after a write
# ---------------------------------------------------------------------------


def test_upsert_warms_the_lexical_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every upsert invalidates the mtime-keyed BM25 cache. The rebuild must
    happen on the writer's thread, so the next search (inside the MCP's
    4-slot search guard) finds it warm."""
    paths = _vault(tmp_path)
    _prepare_search(monkeypatch)
    _seed_index(paths, [("archive/processed/kept.md", 0, _KEPT_TEXT)])
    _write(paths, "knowledge/people/anna-kowalska.md", _NOTE_V1)

    builds: list[int] = []
    import ingest_lib.lexical as lexical_mod

    original = lexical_mod.build_lexical_index

    def counting(texts: list[str]) -> LexicalIndex:
        builds.append(len(texts))
        return original(texts)

    monkeypatch.setattr(lexical_mod, "build_lexical_index", counting)

    upsert_notes(
        paths, ["knowledge/people/anna-kowalska.md"], logger=_LOG, encode=_fake_encode
    )
    assert builds, "upsert did not build the BM25 index"
    before = len(builds)

    semantic.search(paths, _KEPT_TEXT, top_k=3, mode="hybrid")
    assert len(builds) == before, "the search rebuilt the BM25 index"


def test_lexical_build_is_single_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent cold searches build the BM25 index once between them, not
    once each."""
    monkeypatch.setattr(semantic, "_LEXICAL_CACHE", None)
    key: semantic._IndexKey = ("/v/embeddings.npy", "/v/meta.jsonl", 1.0, 2.0)
    rows = [
        semantic.MetaRow(
            source_relative_path="archive/processed/kept.md",
            source_hash="h", title="kept", chunk_idx=0, text=_KEPT_TEXT,
            origin="text", heading_path="", model="m", gen="",
        )
    ]

    import ingest_lib.lexical as lexical_mod

    original = lexical_mod.build_lexical_index
    entered = threading.Event()
    release = threading.Event()
    builds: list[int] = []

    def slow(texts: list[str]) -> LexicalIndex:
        builds.append(len(texts))
        entered.set()
        assert release.wait(10)
        return original(texts)

    monkeypatch.setattr(lexical_mod, "build_lexical_index", slow)

    first = threading.Thread(
        target=lambda: semantic._lexical_for_generation(key, rows)
    )
    first.start()
    assert entered.wait(10)
    second = threading.Thread(
        target=lambda: semantic._lexical_for_generation(key, rows)
    )
    second.start()
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert builds == [1]


# ------------------------------------------------- the embedder seam itself
# AUD-103: CI deliberately runs without torch, so the model is never loaded
# anywhere in the suite and `_query_prefix`, `encode_query`, `_load_embedder`
# and `_autodetect_device` were executed by nothing. A real-model test would
# be skipped by default and off in CI — i.e. it would never run — so these
# pin the same risks (wrong query instruction, an ignored device override, a
# changed encode kwarg) against an injected stand-in instead.

class _RecordingEmbedder:
    """Captures exactly what ``encode_query`` hands the model."""

    def __init__(self) -> None:
        self.sentences: list[list[str]] = []
        self.kwargs: list[dict[str, bool]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        self.sentences.append(list(sentences))
        self.kwargs.append({
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
        })
        return np.ones((len(sentences), _DIM), dtype=np.float64)


def test_encode_query_prepends_the_models_retrieval_instruction(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAIN_QUERY_INSTRUCTION", raising=False)
    model = _RecordingEmbedder()
    vec = semantic.encode_query(model, "how do kalman filters work")

    # The instruction the index was NOT built with must be the query-side
    # prefix, exactly as documented for bge-small-en-v1.5.
    assert model.sentences == [[
        "Represent this sentence for searching relevant passages: "
        "how do kalman filters work"
    ]]
    # Normalised (cosine == dot) and silent: both are load-bearing.
    assert model.kwargs == [{"normalize_embeddings": True, "show_progress_bar": False}]
    assert vec.dtype == np.float32 and vec.shape == (1, _DIM)


def test_query_instruction_can_be_disabled_for_ab_testing(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAIN_QUERY_INSTRUCTION", "0")
    assert semantic._query_prefix() == ""
    model = _RecordingEmbedder()
    semantic.encode_query(model, "kalman filters")
    assert model.sentences == [["kalman filters"]]


def test_query_prefix_is_keyed_to_the_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # A model swap must drop the prefix rather than silently apply the wrong
    # one — the instruction is per-model, not universal.
    monkeypatch.delenv("BRAIN_QUERY_INSTRUCTION", raising=False)
    assert semantic._query_prefix() != ""
    monkeypatch.setattr(semantic, "_MODEL_NAME", "some/other-model")
    assert semantic._query_prefix() == ""


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer."""

    def __init__(self, model_name: str, *, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        return np.ones((len(sentences), _DIM), dtype=np.float32)


def _inject_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import ModuleType
    module = ModuleType("sentence_transformers")
    monkeypatch.setattr(
        module, "SentenceTransformer", _FakeSentenceTransformer, raising=False
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(semantic, "_EMBEDDER_CACHE", None)


def test_load_embedder_honours_the_device_override_and_caches(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_sentence_transformers(monkeypatch)
    monkeypatch.setenv("BRAIN_EMBED_DEVICE", "cpu")

    model, device = semantic._load_embedder()
    assert device == "cpu"
    assert isinstance(model, _FakeSentenceTransformer)
    assert model.model_name == semantic._MODEL_NAME and model.device == "cpu"
    # Cached: the second call must not construct a second model (a cold load
    # is 5-15s of torch init on the real thing).
    again, _ = semantic._load_embedder()
    assert again is model


def test_load_embedder_autodetects_when_no_device_is_set(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_sentence_transformers(monkeypatch)
    monkeypatch.delenv("BRAIN_EMBED_DEVICE", raising=False)
    monkeypatch.setattr(semantic, "_autodetect_device", lambda: "mps")

    _model, device = semantic._load_embedder()
    assert device == "mps"


def test_autodetect_device_falls_back_to_cpu_without_torch(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    real_import = builtins.__import__

    def no_torch(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "torch":
            raise ImportError("no module named torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert semantic._autodetect_device() == "cpu"


def test_hybrid_falling_back_to_lexical_marks_its_hits_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AUD-107: a hybrid query whose embedder fails still answers from the
    BM25 leg, and those hits used to be indistinguishable from a real
    fusion — so a caller could not tell it was reading one leg's ranks."""
    paths = _vault(tmp_path)
    _seed_index(paths, [("archive/processed/a.md", 0, "alpha beta gamma")])
    monkeypatch.setenv("BRAIN_QUERY_INSTRUCTION", "0")

    def boom() -> tuple[object, str]:
        raise RuntimeError("no model here")

    monkeypatch.setattr(semantic, "_load_embedder", boom)
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
    monkeypatch.setattr(semantic, "_LEXICAL_CACHE", None)
    with caplog.at_level(logging.ERROR):
        hits = semantic.search(paths, "alpha", top_k=5, mode="hybrid")

    assert hits, "the lexical leg should still answer"
    assert all(h.degraded for h in hits)
    assert any("DEGRADED" in r.message for r in caplog.records)


def test_a_healthy_hybrid_search_is_not_marked_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("archive/processed/a.md", 0, "alpha beta gamma")])
    _prepare_search(monkeypatch)
    hits = semantic.search(paths, "alpha beta gamma", top_k=5, mode="hybrid")
    assert hits and not any(h.degraded for h in hits)
