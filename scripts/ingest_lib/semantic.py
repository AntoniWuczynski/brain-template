"""Semantic search over ``archive/processed/`` and curated ``knowledge/`` notes.

Canonical-tag matching (the summarizer's ``topics``) is precise but
narrow. Paraphrased or unindexed concepts get missed. This module
complements it: every paragraph chunk — from processed sources and from
hand-written knowledge notes (via ``knowledge.knowledge_records``) — gets
encoded as a vector, the query gets encoded, you sort by cosine similarity
— or, in the default ``hybrid`` mode, by rank fusion of that cosine ranking
with a BM25 one (see ``SearchHit.score``: the fused number is a rank score,
not a similarity).

Implementation choices:

- Embedding model is ``BAAI/bge-small-en-v1.5`` from sentence-transformers.
  Around 100 MB of weights, 384-dim output, fast on CPU and MPS.
  Vectors are L2-normalised so cosine similarity is just a dot product.
- Chunks are greedily packed paragraphs targeting ~400 tokens, no overlap.
- Storage is two files: ``metadata/embeddings.npy`` for the vector
  matrix and ``metadata/embeddings_meta.jsonl`` for the row metadata.
  No database. Row ``i`` of one file only means anything paired with row
  ``i`` of the other, so every meta row carries a ``gen`` stamp derived
  from the vector matrix's contents and a reader that recomputes it from
  the vectors it loaded can tell the two files apart generation by
  generation (see ``_write_index_files``).
- ``build_index`` rebuilds from scratch — fine for thousands of chunks.
  ``upsert_notes`` patches just the rows of a few knowledge notes in
  place (the MCP write path), so a note edit doesn't pay a full rebuild.
- ``status: processed`` records are indexed, and so are the ``partial``
  records whose TEXT is whole and whose loss is visual only (see
  ``_is_indexable``). A ``partial`` whose text is short of the source (the
  pypdf fallback used when MinerU isn't installed — i.e. every PDF on a
  fresh clone — a file truncated at the size cap, a transcript with
  unparsed cues) stays out: ranking half a document as if it were whole is
  worse than a miss. The withheld count is logged loudly at rebuild time so
  an apparently-empty vault is understood as this policy gap, not a missing
  document.
- Writers serialise on an advisory file lock (``metadata/.embeddings.lock``)
  so an MCP-triggered upsert can't interleave with a CLI rebuild.
- Everything is local. Search makes no network calls.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from collections.abc import Callable, Iterator

if TYPE_CHECKING:
    # numpy is imported lazily inside functions (torch-free CLIs shouldn't pay
    # the import cost); this makes the "np.ndarray" annotations resolvable for
    # ruff and mypy without importing it at module load.
    import numpy as np
    from .lexical import LexicalIndex

from . import rank as rank_lib
from .config import RankingConfig, VaultPaths, ranking_config
from .knowledge import _record_for_note, knowledge_records
from .metadata import latest_records_by_path
# AUD-121 split: chunking (pure paragraph-packing over one record's markdown)
# and index-row shapes/lookups (no monkeypatch coupling to this module) moved
# to sibling modules. Re-exported below so every name this module used to
# define stays importable (and, for the mutable/monkeypatched pieces kept
# here, patchable) as ``ingest_lib.semantic.<name>`` — see each sibling's
# docstring for why the rest of this module could not move as freely: nearly
# every remaining function is the target of a direct ``monkeypatch.setattr
# (semantic, "_name", ...)`` in the test suite, or bare-calls another such
# name, and a bare name inside a function is resolved against the globals of
# the module the function is DEFINED in — so patching ``semantic._load_embedder``
# (say) only reaches callers whose def site is this file.
# Every name below is imported under an explicit "as <same name>" alias:
# this module used to DEFINE each of them, so re-exporting them as
# ``ingest_lib.semantic.<name>`` (public and private alike — tests import
# private helpers by name, e.g. ``from ingest_lib.semantic import
# _pack_blocks_with_headings``) is the point, not an accident — the
# redundant alias is pyflakes/ruff's convention for "this import is an
# intentional re-export", so F401 does not fire on it.
from .semantic_chunking import (
    Chunk as Chunk,
    _HEADER_BLOCK_RE as _HEADER_BLOCK_RE,
    _HEADING_LINE_RE as _HEADING_LINE_RE,
    _MIN_CHARS as _MIN_CHARS,
    _TARGET_CHARS as _TARGET_CHARS,
    _TARGET_TOKENS as _TARGET_TOKENS,
    _chunks_for_record as _chunks_for_record,
    _is_lone_heading as _is_lone_heading,
    _pack_blocks_with_headings as _pack_blocks_with_headings,
    _split_into_blocks as _split_into_blocks,
    _split_oversize_block as _split_oversize_block,
    _strip_processed_footer as _strip_processed_footer,
    _strip_processed_header as _strip_processed_header,
    _update_heading_stack as _update_heading_stack,
    chunk_markdown as chunk_markdown,
)
from .semantic_meta import (
    MetaRow as MetaRow,
    SearchHit as SearchHit,
    _VISUALS_ONLY_PARTIAL_EXTRACTORS as _VISUALS_ONLY_PARTIAL_EXTRACTORS,
    _by_extension as _by_extension,
    _embed_model_tag as _embed_model_tag,
    _embed_text as _embed_text,
    _hit_from_row as _hit_from_row,
    _is_indexable as _is_indexable,
    _meta_row as _meta_row,
    _reextract_advice as _reextract_advice,
    _row_from_json as _row_from_json,
    chunks_for_source as chunks_for_source,
)
# ``_MODEL_NAME`` lives in ``semantic_model`` rather than being defined here,
# because ``semantic_meta._embed_model_tag``/``_embed_text`` need the SAME
# name and a sibling may not import this module (see the block comment
# above) — so the one name both sides need sits lower than either.
from .semantic_model import _MODEL_NAME as _MODEL_NAME


# Soft tripwire: below ~50k chunks the flat npy + JSONL store is fine; past
# it a linear matmul per query starts to bite (see TODO.md: migrate to
# sqlite-vec). Warn at 80% so there's runway before it matters.
_SQLITE_VEC_HINT_THRESHOLD = 40_000


def _warn_if_index_large(chunk_count: int, logger: logging.Logger) -> None:
    if chunk_count >= _SQLITE_VEC_HINT_THRESHOLD:
        logger.warning(
            "semantic: %d chunks indexed — approaching the ~50k point where the "
            "flat npy+JSONL store gets slow. Consider migrating to sqlite-vec "
            "(TODO.md).", chunk_count,
        )

# Per-model retrieval query instruction. BGE v1.5 is trained to prepend this
# to the QUERY only (passages are embedded raw), which is exactly how the
# index is built — so this is a query-side-only change that needs NO reindex.
# Keyed to _MODEL_NAME so a future model swap can't silently apply the wrong
# prefix. Disable with BRAIN_QUERY_INSTRUCTION=0 (e.g. to A/B via the eval
# harness).
_QUERY_INSTRUCTIONS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def _query_prefix() -> str:
    if os.environ.get("BRAIN_QUERY_INSTRUCTION", "1") == "0":
        return ""
    return _QUERY_INSTRUCTIONS.get(_MODEL_NAME, "")


class Embedder(Protocol):
    """Structural type for the lazily-loaded sentence-transformers model —
    only the ``encode`` method this module actually calls."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
        batch_size: int = ...,
    ) -> np.ndarray: ...


def encode_query(model: Embedder, query: str) -> np.ndarray:
    """Encode a search query with the model's documented retrieval instruction
    (query-side only). Shared by ``search`` and ``describe._retrieve`` so both
    query the index the same way."""
    import numpy as np

    return np.asarray(
        model.encode(
            [_query_prefix() + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

# Process-wide caches. Cold load is 5-15 seconds (model weights + torch
# init); without these every CLI search and every MCP request would pay
# that cost. The index cache uses mtime so the next search sees ingest's
# rewrite automatically.
import threading as _threading
_CACHE_LOCK = _threading.Lock()
_EMBEDDER_CACHE: tuple[Embedder, str] | None = None
# key is (vectors_path, meta_path, vectors_mtime, meta_mtime): the mtimes
# stop a search caching a vectors file from one rebuild against a meta file
# from another, and the paths stop two vaults in one process (the pytest
# suite) serving each other's index on an mtime collision.
_IndexKey = tuple[str, str, float, float]
_INDEX_CACHE: tuple[_IndexKey, np.ndarray, list[MetaRow]] | None = None
# BM25 index built from the SAME generation of meta as _INDEX_CACHE, keyed
# identically. Building it from the already-loaded meta (not a second,
# independently-mtime'd read) guarantees vectors, meta and lexical rows all
# come from one rebuild — a reindex between two separate loads used to pair
# generation-A meta with generation-B lexical rows (IndexError / wrong hits).
_LEXICAL_CACHE: tuple[_IndexKey, LexicalIndex] | None = None
# Held only while BUILDING the BM25 index, so the up-to-4 concurrent cold
# searches behind the MCP search guard build it once between them instead of
# once each (~0.8 s over 13k rows). Never held while reading the cache.
_LEXICAL_BUILD_LOCK = _threading.Lock()


@contextmanager
def _index_lock(paths: VaultPaths) -> Iterator[None]:
    """Cross-process advisory lock around index writes.

    ``build_index`` historically had no file lock; an MCP-triggered upsert
    racing a CLI ingest could interleave the two ``os.replace`` calls and
    pair vectors from one run with meta from another. ``flock`` is advisory
    and per open-file-description, so this only guards writers that take it
    — readers stay lock-free (a full rebuild holds this lock for minutes;
    blocking every search behind it would be worse than the race) and rely
    on the per-row generation stamp checked in ``_load_index`` to reject a
    mismatched pair.

    NOT reentrant: a second ``open()`` + ``LOCK_EX`` in the same process
    blocks on the first. Never call a lock-taking function (``build_index``)
    while holding it — fall back *after* the ``with`` block exits.
    """
    lock_path = paths.metadata / ".embeddings.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _load_embedder() -> tuple[Embedder, str]:
    # Imported lazily because sentence-transformers pulls in torch and
    # transformers, and we don't want that cost on every CLI invocation.
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE
    from sentence_transformers import SentenceTransformer

    with _CACHE_LOCK:
        if _EMBEDDER_CACHE is None:
            device = os.environ.get("BRAIN_EMBED_DEVICE") or _autodetect_device()
            model = SentenceTransformer(_MODEL_NAME, device=device)
            _EMBEDDER_CACHE = (model, device)
    return _EMBEDDER_CACHE


def _vectors_stamp(vectors: np.ndarray) -> str:
    """Generation id for one vector matrix, derived from its contents.

    ``.npy`` has no room for metadata of its own, so the stamp travels in
    the meta rows instead (``_write_index_files``) and a reader recomputes
    it from the vectors it just loaded. Two files from different
    generations therefore never pair, whatever their row counts."""
    import numpy as np

    buf = np.ascontiguousarray(vectors)
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{buf.dtype.str}:{buf.shape}:".encode("ascii"))
    h.update(buf.tobytes())
    return h.hexdigest()


def _pair_is_consistent(vectors: np.ndarray, meta: list[MetaRow]) -> bool:
    """Are these vectors and these meta rows the same index generation?

    Equal row counts do NOT answer that: an upsert that replaces a note
    with an equally-chunked version keeps the count identical while moving
    every row that followed it, so vectors from after the upsert paired
    with meta from before it line up in size and disagree in content. The
    generation stamp does answer it. Rows written before the stamp existed
    carry ``""`` and are accepted on their row count alone."""
    if len(meta) != vectors.shape[0]:
        return False
    stamp = meta[0]["gen"] if meta else ""
    return not stamp or stamp == _vectors_stamp(vectors)


def _load_index(
    vectors_path: Path, meta_path: Path
) -> tuple[_IndexKey, np.ndarray, list[MetaRow]] | None:
    """Load the .npy vectors and .jsonl metadata, cached by mtime.

    Cheap to call repeatedly; the actual disk read only fires when either
    index file changes (i.e. ingest rewrote it). Readers take no lock, so a
    read CAN land between the writer's two renames; the generation stamp
    rejects that pair and the load is retried once, because the window is
    two syscalls wide and the retry lands after it. A pair still mismatched
    on the second attempt is a torn index (a crash between the renames): it
    is refused and NOT cached, so search says "rebuild required" instead of
    serving rows attributed to the wrong source.
    """
    import numpy as np

    global _INDEX_CACHE
    log = logging.getLogger(__name__)
    for attempt in (1, 2):
        try:
            key = (
                str(vectors_path), str(meta_path),
                vectors_path.stat().st_mtime, meta_path.stat().st_mtime,
            )
        except OSError:
            return None
        # Snapshot the global into a local before reading its fields: a
        # concurrent rebuild can reassign _INDEX_CACHE between the [1] and
        # [2] reads, pairing vectors from one generation with meta from
        # another.
        cache = _INDEX_CACHE
        if cache is not None and cache[0] == key:
            return key, cache[1], cache[2]
        vectors = np.load(vectors_path)
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = [
                _row_from_json(json.loads(ln), i)
                for i, ln in enumerate(fh) if ln.strip()
            ]
        if _pair_is_consistent(vectors, meta):
            with _CACHE_LOCK:
                _INDEX_CACHE = (key, vectors, meta)
            return key, vectors, meta
        if attempt == 1:
            log.warning(
                "semantic: index pair from two generations (vectors=%d rows, "
                "meta=%d rows) — a concurrent rewrite? re-reading",
                vectors.shape[0], len(meta),
            )
    log.error(
        "semantic: torn index — embeddings.npy and embeddings_meta.jsonl are "
        "from different generations (a crash between the two renames). Rebuild "
        "with 'uv run python scripts/ingest.py --rebuild-search-index'"
    )
    return None


def _lexical_for_generation(key: _IndexKey, meta: list[MetaRow]) -> LexicalIndex:
    """BM25 index over the ALREADY-LOADED meta rows, cached under the same
    ``(vectors_mtime, meta_mtime)`` key as ``_load_index``. Building it here
    (not via a second, independently-mtime'd read of the meta file) keeps
    vectors, meta and lexical rows on one generation."""
    from . import lexical as _lex

    global _LEXICAL_CACHE
    cache = _LEXICAL_CACHE
    if cache is not None and cache[0] == key:
        return cache[1]
    with _LEXICAL_BUILD_LOCK:
        # Re-check: whoever held the lock has just built this generation.
        cache = _LEXICAL_CACHE
        if cache is not None and cache[0] == key:
            return cache[1]
        index = _lex.build_lexical_index([m["text"] for m in meta])
        with _CACHE_LOCK:
            _LEXICAL_CACHE = (key, index)
    return index


def _autodetect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so renames inside it survive a power loss."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_index_files(
    vectors: np.ndarray, rows: list[MetaRow], *, vectors_path: Path, meta_path: Path
) -> None:
    """Publish one generation of the index pair: both temp files written
    and fsynced first, then renamed, then the containing directory fsynced
    (a rename is only durable once the directory entry is on disk).

    Every meta row is stamped with ``_vectors_stamp(vectors)``, so the two
    files name the generation they belong to. The swap is still two
    renames, and a reader can land between them just as a crash can — in
    both cases the stamp in the meta rows no longer matches the vectors on
    disk, and the pair is refused (``_load_index``) or rebuilt
    (``upsert_notes``) rather than serving rows attributed to the wrong
    source. Recovery after a crash is a full rebuild:
    ``uv run python scripts/ingest.py --rebuild-search-index``."""
    import numpy as np

    stamp = _vectors_stamp(vectors)
    tmp_vec = vectors_path.with_name(vectors_path.stem + ".tmp.npy")
    with tmp_vec.open("wb") as vec_fh:
        np.save(vec_fh, vectors)
        vec_fh.flush()
        os.fsync(vec_fh.fileno())

    tmp_meta = meta_path.with_suffix(".jsonl.tmp")
    with tmp_meta.open("w", encoding="utf-8") as meta_fh:
        for row in rows:
            stamped: MetaRow = row.copy()
            stamped["gen"] = stamp
            meta_fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        meta_fh.flush()
        os.fsync(meta_fh.fileno())

    os.replace(tmp_vec, vectors_path)
    os.replace(tmp_meta, meta_path)
    _fsync_dir(meta_path.parent)


def build_index(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
) -> int:
    """(Re)build the dense vector index from scratch. Returns chunk count."""
    paths.ensure()
    # The whole read-compute-write cycle holds the writer lock: a per-note
    # upsert reading the files mid-rebuild would otherwise base its patch
    # on rows this rebuild is about to throw away.
    with _index_lock(paths):
        # Ingested sources + curated knowledge/ notes (their processed_path
        # is the note itself, so they chunk and embed like any processed
        # markdown).
        candidates = (
            list(latest_records_by_path(paths.metadata_index_jsonl).values())
            + knowledge_records(paths)
        )
        records = [r for r in candidates if _is_indexable(r)]
        # Surface the withheld count loudly so a vault that seems to "have
        # nothing about X" is understood as an indexing-policy gap, not an
        # empty vault.
        withheld = [
            r for r in candidates
            if r.status == "partial" and r.processed_path and not _is_indexable(r)
        ]
        if withheld:
            logger.warning(
                "semantic: %d 'partial' record(s) are NOT indexed (not "
                "searchable) — their extracted TEXT is short of the source, so "
                "indexing them would rank half a document as if it were whole. "
                "By extension: %s. %s",
                len(withheld),
                _by_extension(withheld),
                _reextract_advice(withheld),
            )
        if not records:
            logger.info("semantic: no indexable records to index")
            return 0

        chunks = [c for rec in records for c in _chunks_for_record(rec, paths)]
        if not chunks:
            logger.info("semantic: 0 chunks after chunking")
            return 0

        logger.info(
            "semantic: encoding %d chunk(s) from %d source(s)",
            len(chunks), len(records),
        )
        _warn_if_index_large(len(chunks), logger)

        try:
            import numpy as np
        except ImportError as exc:
            logger.warning("semantic: numpy not installed (%s) — skipping", exc)
            return 0

        try:
            model, device = _load_embedder()
        except ImportError as exc:
            logger.warning(
                "semantic: sentence-transformers not installed (%s) — skipping. "
                "Install with: uv pip install sentence-transformers",
                exc,
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — model load can fail many ways
            logger.warning("semantic: model load failed (%r) — skipping", exc)
            return 0

        logger.info("semantic: model %s on %s", _MODEL_NAME, device)

        texts = [_embed_text(c) for c in chunks]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        vectors_path = paths.metadata / "embeddings.npy"
        meta_path = paths.metadata / "embeddings_meta.jsonl"
        _write_index_files(
            vectors,
            [_meta_row(c) for c in chunks],
            vectors_path=vectors_path,
            meta_path=meta_path,
        )

        logger.info(
            "semantic: wrote %d vectors (dim=%d) to %s",
            vectors.shape[0],
            vectors.shape[1],
            vectors_path.relative_to(paths.root),
        )
        return len(chunks)


# ---------------------------------------------------------------------------
# Incremental upsert
# ---------------------------------------------------------------------------

# Mirrors knowledge.KNOWLEDGE_NOTE_DIRS plus the memory areas (meetings/,
# assistant/) the MCP server writes into — keep in sync by hand. Kept as a
# local tuple rather than importing the constant: that tuple is being
# extended in the same change-set, and the upsert allowlist must not
# silently widen or narrow underneath us if the two ever diverge.
# knowledge/index/ and knowledge/concepts/ stay excluded for the same
# reason they're excluded from knowledge_records(): they're generated
# from sources, and embedding them would double-count archive content.
_UPSERTABLE_DIRS: tuple[str, ...] = (
    "notes",
    "organisations",
    "people",
    "projects",
    "chats",
    "personal",
    "research",
    "university",
    "meetings",
    "assistant",
)


def upsert_notes(
    paths: VaultPaths,
    note_rel_paths: list[str],
    *,
    logger: logging.Logger,
    encode: Callable[[list[str]], np.ndarray] | None = None,
) -> int:
    """Patch the index rows for a few knowledge notes without a full
    rebuild. Returns the number of chunks newly written (0 when every
    path was skipped or only deletions happened).

    For each vault-root-relative path: a missing/empty note has its rows
    dropped; otherwise its old rows are replaced by freshly encoded ones.
    Rows of untouched sources are carried over verbatim. ``encode``
    overrides the embedding model in tests.

    Honesty note: encoding a single note can differ from the same text
    encoded inside a full-rebuild batch at float-ulp level (batch padding,
    kernel selection). The periodic full ``build_index`` is the
    consistency pass; upsert keeps search fresh between passes.
    """
    vectors_path = paths.metadata / "embeddings.npy"
    meta_path = paths.metadata / "embeddings_meta.jsonl"

    valid: list[str] = []
    seen: set[str] = set()
    for rel in note_rel_paths:
        if rel in seen:
            continue
        seen.add(rel)
        if rel.startswith("knowledge/assistant/archive/"):
            # Mirror knowledge.scan_knowledge's exclusion (see
            # knowledge._ASSISTANT_ARCHIVE_PREFIX): an archived promoted fact
            # already lives in the entity note it was promoted into (or in a
            # digest), so indexing it would make that fact double-retrievable.
            # A consolidate CLI handing us an archive destination is a no-op.
            logger.info(
                "semantic: upsert skipped %s — assistant archive (historical "
                "record, already indexed via its entity note)", rel,
            )
            continue
        if any(rel.startswith(f"knowledge/{d}/") for d in _UPSERTABLE_DIRS):
            valid.append(rel)
        else:
            logger.warning(
                "semantic: upsert skipped %s — not under a hand-edited "
                "knowledge area", rel,
            )
    valid.sort()
    if not valid:
        return 0

    if not vectors_path.exists() or not meta_path.exists():
        # A fresh vault has no index. Creating a one-note index here would
        # mask the 'never built' state for every other source, so build
        # the real thing instead.
        logger.info("semantic: no index yet — running a full build instead of upsert")
        return build_index(paths, logger=logger)

    try:
        import numpy as np
    except ImportError as exc:
        logger.warning("semantic: numpy not installed (%s) — skipping upsert", exc)
        return 0

    # The fallback rebuild happens AFTER the lock is released: build_index
    # takes the same (non-reentrant) lock, and the tiny unlock window is
    # harmless because a full rebuild reads index.jsonl + the notes, never
    # the files another writer might touch in between.
    needs_rebuild = False
    written = 0
    # (key, rows) of the generation this call wrote, for the BM25 warm-up
    # after the lock is released. The key is stat'd UNDER the lock so it can
    # only ever name our own write.
    warm: tuple[_IndexKey, list[MetaRow]] | None = None
    with _index_lock(paths):
        # Fresh read under the lock — deliberately not _load_index, whose
        # process-wide cache would serve (and then retain) pre-upsert rows.
        meta: list[MetaRow]
        try:
            vectors = np.load(vectors_path)
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = [
                    _row_from_json(json.loads(ln), i)
                    for i, ln in enumerate(fh) if ln.strip()
                ]
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            logger.warning("semantic: failed to load index (%r) — full rebuild", exc)
            needs_rebuild = True
            meta, vectors = [], None

        if not needs_rebuild and not _pair_is_consistent(vectors, meta):
            # Either a size mismatch or — the case a size check cannot see —
            # vectors and meta left behind by a crash between the two renames
            # of an earlier write. Patching rows into that pair would keep
            # serving every row after the edited note under the wrong source.
            logger.warning(
                "semantic: index pair inconsistent (vectors=%d rows, meta=%d rows) "
                "— full rebuild", vectors.shape[0], len(meta),
            )
            needs_rebuild = True

        # If the index was built in a DIFFERENT vector space — another
        # embedding model, or the same model with a different
        # BRAIN_EMBED_HEADING_CONTEXT setting — patching new rows in would
        # silently mix the two (the dim check below only catches a dimension
        # change). The per-row ``model`` tag exists precisely for this guard.
        want_model = _embed_model_tag()
        if not needs_rebuild and any(
            m["model"] not in ("", want_model) for m in meta
        ):
            logger.warning(
                "semantic: index has rows from a different model (want %s) — full rebuild",
                want_model,
            )
            needs_rebuild = True

        if not needs_rebuild:
            new_chunks: list[Chunk] = []
            updated: set[str] = set()
            for rel in valid:  # sorted; _chunks_for_record keeps chunk_idx ascending
                updated.add(rel)
                md = paths.root / rel
                if not md.is_file():
                    continue  # deleted note: rows dropped, nothing re-added
                try:
                    data = md.read_bytes()
                    rec = _record_for_note(md, paths)
                except FileNotFoundError:
                    continue  # vanished between is_file and read: genuinely gone
                except OSError as exc:
                    # Unreadable right now is NOT deleted (mirrors
                    # knowledge.scan_knowledge): keep the old rows.
                    logger.warning("semantic: upsert could not read %s (%s)", rel, exc)
                    updated.discard(rel)
                    continue
                if rec is None:
                    continue  # empty/whitespace note: rows dropped
                # Hash and chunk the SAME bytes. _record_for_note hashes its
                # own read and _chunks_for_record would read a third time, so
                # an edit landing between them (Obsidian and consolidate.py
                # are outside the MCP write lock) produced a row whose
                # source_hash named content its text did not hold — sweep saw
                # no drift and search served the stale snippet forever.
                rec = replace(rec, source_hash=hashlib.sha256(data).hexdigest())
                new_chunks.extend(
                    _chunks_for_record(
                        rec, paths, text=data.decode("utf-8", errors="replace")
                    )
                )

            new_vecs = None
            if new_chunks:
                texts = [_embed_text(c) for c in new_chunks]
                if encode is not None:
                    new_vecs = np.asarray(encode(texts), dtype=np.float32)
                else:
                    try:
                        model, _ = _load_embedder()
                    except Exception as exc:  # noqa: BLE001 — model load can fail many ways
                        logger.warning(
                            "semantic: model load failed (%r) — upsert skipped", exc
                        )
                        return 0
                    new_vecs = np.asarray(
                        model.encode(
                            texts,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=32,
                        ),
                        dtype=np.float32,
                    )
                if vectors.shape[0] and new_vecs.shape[1] != vectors.shape[1]:
                    # Dimension drift means the index was built by a
                    # different model — patching rows in would corrupt it.
                    logger.warning(
                        "semantic: embedding dim mismatch (index=%d, new=%d) — full rebuild",
                        vectors.shape[1], new_vecs.shape[1],
                    )
                    needs_rebuild = True

            if not needs_rebuild:
                keep = [
                    i for i, m in enumerate(meta)
                    if m["source_relative_path"] not in updated
                ]
                kept_vecs = vectors[keep]
                rows = [meta[i] for i in keep]
                if new_chunks:
                    assert new_vecs is not None  # set above whenever new_chunks is non-empty
                    all_vecs = (
                        np.concatenate([kept_vecs, new_vecs], axis=0)
                        if kept_vecs.shape[0] else new_vecs
                    )
                    rows.extend(_meta_row(c) for c in new_chunks)
                else:
                    all_vecs = kept_vecs
                _write_index_files(
                    all_vecs, rows, vectors_path=vectors_path, meta_path=meta_path
                )
                written = len(new_chunks)
                warm = (
                    (
                        str(vectors_path), str(meta_path),
                        vectors_path.stat().st_mtime, meta_path.stat().st_mtime,
                    ),
                    rows,
                )
                logger.info(
                    "semantic: upsert wrote %d chunk(s) for %d note(s), %d row(s) total",
                    written, len(valid), len(rows),
                )
                # The MCP write path is the one that grows the index between
                # full rebuilds, so the sqlite-vec tripwire has to fire here
                # too — build_index alone would only ever see it after a
                # rebuild the operator has no reason to run yet.
                _warn_if_index_large(len(rows), logger)

    if needs_rebuild:
        return build_index(paths, logger=logger)
    if warm is not None:
        # Warm the BM25 cache for the generation just written. Every upsert
        # invalidates it (the key carries both index files' mtimes), so
        # without this the next search pays the rebuild inside the MCP's
        # 4-slot search guard; upsert_notes already runs on the background
        # refresher thread, which is where that second is affordable. Done
        # outside the index lock so a CLI rebuild waiting on it doesn't also
        # wait on the build.
        _lexical_for_generation(*warm)
    return written


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_VALID_MODES: tuple[str, ...] = ("dense", "lexical", "hybrid")
_RRF_K = 60          # reciprocal-rank-fusion damping (standard)
_CANDIDATES = 100    # per-ranking candidate pool feeding the fusion


def search(
    paths: VaultPaths,
    query: str,
    *,
    top_k: int = 10,
    mode: str = "hybrid",
    logger: logging.Logger | None = None,
) -> list[SearchHit]:
    """Retrieve the top-k chunks for ``query``.

    ``mode``:
    - ``dense``  — embedding cosine only (paraphrase-friendly).
    - ``lexical``— BM25 over the chunk texts only (exact identifiers; needs
      NO embedding model, so it works on a machine without the weights).
    - ``hybrid`` (default) — reciprocal-rank fusion of both, which fixes the
      class of exact-match failures dense-only cannot.
    """
    log = logger or logging.getLogger(__name__)
    if mode not in _VALID_MODES:
        log.warning("semantic: unknown mode %r — using hybrid", mode)
        mode = "hybrid"
    vectors_path = paths.metadata / "embeddings.npy"
    meta_path = paths.metadata / "embeddings_meta.jsonl"

    if not vectors_path.exists() or not meta_path.exists():
        log.warning(
            "semantic: no index yet — run 'uv run python scripts/ingest.py "
            "--rebuild-search-index' first"
        )
        return []

    try:
        import numpy as np
    except ImportError as exc:
        log.warning("semantic: numpy not installed (%s)", exc)
        return []

    loaded = _load_index(vectors_path, meta_path)
    if loaded is None:
        # Either the files vanished between the exists() check and the load,
        # or the pair is torn — _load_index logged which.
        log.warning("semantic: no usable index (see the load error above)")
        return []
    key, vectors, meta = loaded
    if len(meta) != vectors.shape[0]:
        log.warning(
            "semantic: index size mismatch (vectors=%d, meta=%d) — rebuild required",
            vectors.shape[0], len(meta),
        )
        return []
    n = len(meta)
    # Widen the candidate pool when top_k exceeds it, so a large request (e.g.
    # recency.memory_search's fetch_k=500 filtered path) isn't silently
    # truncated to 100. Behaviour-preserving for the MCP callers (top_k<=50).
    cand = min(n, max(_CANDIDATES, top_k))

    # Optional ranking layers, all OFF unless the environment switches them
    # on (config.ranking_config) and all hybrid-only: dense and lexical
    # return raw cosine / BM25 scores, whose meaning an additive boost or a
    # cross-encoder reorder would silently change.
    cfg = ranking_config() if mode == "hybrid" else RankingConfig()

    # --- dense ranking (skipped for lexical mode -> no model load) ----------
    dense_rank: list[int] = []
    dense_scores = None
    degraded = False
    if mode in ("dense", "hybrid"):
        try:
            model, _ = _load_embedder()
        except Exception as exc:  # noqa: BLE001
            if mode == "dense":
                log.warning("semantic: model load failed (%r)", exc)
                return []
            # A hybrid caller that silently receives BM25-only hits cannot
            # tell them from a real hybrid result, so say so loudly enough
            # that an operator sees it, and record it on the hits below.
            log.error(
                "semantic: model load failed (%r) — DEGRADED to lexical-only "
                "results for this query; hits are BM25 ranks, not hybrid", exc
            )
            mode = "lexical"
            degraded = True
        if mode != "lexical":
            q_vec = encode_query(model, query)
            dense_scores = (vectors @ q_vec.T).ravel()
            top = np.argpartition(-dense_scores, cand - 1)[:cand]
            dense_rank = [int(i) for i in top[np.argsort(-dense_scores[top])]]

    # AFTER the model load: a degraded (lexical-only) search returns below
    # without ever fusing the paraphrases, so asking for them first paid an
    # LLM call for work that was thrown away, and logged it as a success.
    variants: list[str] = []
    if cfg.query_expansion and not degraded:
        from . import rank as _rank

        variants = _rank.expand_query(paths, query, cfg, log)
        if variants:
            log.info("semantic: query expanded with %d paraphrase(s)", len(variants))

    # --- lexical ranking ----------------------------------------------------
    lex_scores: dict[int, float] = {}
    lex_rank: list[int] = []
    if mode in ("lexical", "hybrid"):
        from . import lexical as _lex
        # Build BM25 from the meta we already loaded (same generation), not a
        # second independently-mtime'd read that a concurrent reindex could
        # desync from `meta`.
        lidx = _lexical_for_generation(key, meta)
        lex_scores = _lex.score(lidx, query)
        lex_rank = [i for i, _s in sorted(
            lex_scores.items(), key=lambda kv: (-kv[1], kv[0]))][:cand]

    # --- fuse ---------------------------------------------------------------
    if mode == "dense":
        # dense mode always populated dense_scores above (it only stays None
        # when the model load failed, which already returned) — assert so the
        # indexing below is well-typed rather than "Any | None".
        assert dense_scores is not None
        return [_hit_from_row(meta, i, float(dense_scores[i])) for i in dense_rank[:top_k]]
    if mode == "lexical":
        return [
            _hit_from_row(meta, i, float(lex_scores.get(i, 0.0)), degraded=degraded)
            for i in lex_rank[:top_k]
        ]

    # hybrid: reciprocal-rank fusion of the two candidate lists.
    rrf: dict[int, float] = {}

    def _fuse(order: list[int], weight: float) -> None:
        for rank, i in enumerate(order, start=1):
            rrf[i] = rrf.get(i, 0.0) + weight / (_RRF_K + rank)

    _fuse(dense_rank, 1.0)
    _fuse(lex_rank, 1.0)
    # Query expansion: each paraphrase contributes its own dense and lexical
    # rankings at a discount, so a paraphrase can lift a source the original
    # wording missed without being able to outvote the query actually asked.
    for variant in variants:
        v_vec = encode_query(model, variant)
        v_scores = (vectors @ v_vec.T).ravel()
        v_top = np.argpartition(-v_scores, cand - 1)[:cand]
        _fuse(
            [int(i) for i in v_top[np.argsort(-v_scores[v_top])]],
            cfg.query_expansion_weight,
        )
        v_lex = _lex.score(lidx, variant)
        _fuse(
            [i for i, _s in sorted(v_lex.items(), key=lambda kv: (-kv[1], kv[0]))][:cand],
            cfg.query_expansion_weight,
        )

    fused_all = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))
    if cfg.graph_boost or cfg.salience or cfg.rerank:
        fused_all = rank_lib.apply_layers(
            paths, query, fused_all, meta, cfg=cfg, log=log, unit=1.0 / (_RRF_K + 1)
        )
    return [_hit_from_row(meta, i, s) for i, s in fused_all[:top_k]]
