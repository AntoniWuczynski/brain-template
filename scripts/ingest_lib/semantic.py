"""Semantic search over ``archive/processed/`` and curated ``knowledge/`` notes.

Canonical-tag matching (the summarizer's ``topics``) is precise but
narrow. Paraphrased or unindexed concepts get missed. This module
complements it: every paragraph chunk — from processed sources and from
hand-written knowledge notes (via ``knowledge.knowledge_records``) — gets
encoded as a vector, the query gets encoded, you sort by cosine similarity.

Implementation choices:

- Embedding model is ``BAAI/bge-small-en-v1.5`` from sentence-transformers.
  Around 100 MB of weights, 384-dim output, fast on CPU and MPS.
  Vectors are L2-normalised so cosine similarity is just a dot product.
- Chunks are greedily packed paragraphs targeting ~400 tokens, no overlap.
- Storage is two files: ``metadata/embeddings.npy`` for the vector
  matrix and ``metadata/embeddings_meta.jsonl`` for the row metadata.
  No database.
- ``build_index`` rebuilds from scratch — fine for thousands of chunks.
  ``upsert_notes`` patches just the rows of a few knowledge notes in
  place (the MCP write path), so a note edit doesn't pay a full rebuild.
- ONLY ``status: processed`` records are indexed. ``partial`` notes (the
  pypdf fallback used when MinerU isn't installed — i.e. every PDF on a
  fresh clone) are deliberately NOT searchable; the count is logged loudly
  at rebuild time so an apparently-empty vault is understood as this policy
  gap, not a missing document. Install MinerU and re-ingest for full,
  searchable PDF extraction.
- Writers serialise on an advisory file lock (``metadata/.embeddings.lock``)
  so an MCP-triggered upsert can't interleave with a CLI rebuild.
- Everything is local. Search makes no network calls.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from .config import VaultPaths
from .knowledge import KNOWLEDGE_EXTRACTOR, _record_for_note, knowledge_records
from .metadata import IndexRecord, latest_records_by_path
from .notes import _split_frontmatter


_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_TARGET_TOKENS = 400          # rough budget per chunk; English ≈ chars/4
_TARGET_CHARS = _TARGET_TOKENS * 4
_MIN_CHARS = 80               # skip tiny chunks (headings on their own line)


@dataclass(frozen=True)
class Chunk:
    """One chunk to be embedded."""
    source_relative_path: str   # the original raw path
    source_hash: str
    title: str
    chunk_idx: int              # 0-based, within the source
    text: str
    origin: str = ""            # the record's extractor (e.g. "knowledge-note")


@dataclass(frozen=True)
class SearchHit:
    """One search result."""
    score: float                # cosine similarity, in [-1, 1] (usually 0..1)
    source_relative_path: str
    title: str
    chunk_idx: int
    snippet: str                # the actual chunk text
    origin: str = ""            # "" on indexes built before origin existed


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_HEADER_BLOCK_RE = re.compile(
    r"\A(?:#\s.*?\n+>\s.*?\n(?:>\s.*?\n)*\n+---\n+)",
    re.DOTALL,
)


def _strip_processed_header(text: str) -> str:
    """Drop non-content wrappers before chunking: a leading YAML
    frontmatter fence (knowledge notes), the title + metadata block that
    ``write_processed_note`` prepends, AND the trailing ``---`` +
    ``## Processing notes`` footer it appends (processed notes). Embedding
    any of them would pollute search with metadata — e.g. a verbatim MinerU
    error in the footer would surface for a query like "CUDA out of memory".

    Frontmatter detection MUST agree with ``notes._split_frontmatter``
    (the parser knowledge.py uses to extract topics): it tolerates
    fence whitespace and refuses to strip unless the block yaml-parses
    to a mapping — so a leading ``---`` horizontal rule never swallows
    body content the way a bare regex scan would."""
    _, text = _split_frontmatter(text)
    m = _HEADER_BLOCK_RE.match(text)
    if m:
        text = text[m.end():]
    return _strip_processed_footer(text)


def _strip_processed_footer(text: str) -> str:
    """Remove the trailing ``---`` + ``## Processing notes`` block that
    ``write_processed_note`` appends. Anchored on the LAST such heading
    preceded by a fence, so ordinary body content is never touched."""
    lines = text.splitlines(keepends=True)
    for j in range(len(lines) - 1, -1, -1):
        if lines[j].strip() == "## Processing notes":
            k = j - 1
            while k >= 0 and lines[k].strip() == "":
                k -= 1
            if k >= 0 and lines[k].strip() == "---":
                return "".join(lines[:k]).rstrip() + "\n"
            break
    return text


def _split_into_blocks(text: str) -> list[str]:
    """Paragraph-level split on blank lines."""
    blocks = re.split(r"\n\s*\n+", text)
    return [b.strip() for b in blocks if b.strip()]


def chunk_markdown(text: str, *, min_chars: int = _MIN_CHARS) -> list[str]:
    """Greedy-pack paragraphs into ~``_TARGET_CHARS`` chunks.

    ``min_chars`` is the floor below which a chunk is dropped as noise.
    Curated knowledge notes pass a lower floor (see ``_chunks_for_record``):
    a one-line memory-fact note is legitimately short, and dropping it would
    make the fact permanently unsearchable AND permanently flag it as
    index drift in the sweep."""
    blocks = _split_into_blocks(_strip_processed_header(text))
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for block in blocks:
        block_len = len(block)
        if buf and buf_len + block_len + 2 > _TARGET_CHARS:
            chunks.append("\n\n".join(buf))
            buf = [block]
            buf_len = block_len
        elif block_len > _TARGET_CHARS:
            # Block is bigger than the budget. Flush what we've got and
            # emit the oversize block on its own. Don't split a paragraph
            # mid-sentence: things like extracted tables arrive as one
            # giant block and splitting them hurts retrieval.
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_len = [], 0
            chunks.append(block)
        else:
            buf.append(block)
            buf_len += block_len + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if len(c) >= min_chars]


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

# Process-wide caches. Cold load is 5-15 seconds (model weights + torch
# init); without these every CLI search and every MCP request would pay
# that cost. The index cache uses mtime so the next search sees ingest's
# rewrite automatically.
import threading as _threading
_CACHE_LOCK = _threading.Lock()
_EMBEDDER_CACHE: tuple[object, str] | None = None
# key is (vectors_mtime, meta_mtime) so a search can't cache a vectors
# file from one rebuild against a meta file from another.
_INDEX_CACHE: tuple[tuple[float, float], "object", list[dict]] | None = None


@contextmanager
def _index_lock(paths: VaultPaths) -> Iterator[None]:
    """Cross-process advisory lock around index writes.

    ``build_index`` historically had no file lock; an MCP-triggered upsert
    racing a CLI ingest could interleave the two ``os.replace`` calls and
    pair vectors from one run with meta from another. ``flock`` is advisory
    and per open-file-description, so this only guards writers that take it
    — readers stay lock-free and rely on the mtime-keyed ``_load_index``
    cache plus search's row-count check to reject a torn pair.

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


def _load_embedder():
    # Imported lazily because sentence-transformers pulls in torch and
    # transformers, and we don't want that cost on every CLI invocation.
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    with _CACHE_LOCK:
        if _EMBEDDER_CACHE is None:
            device = os.environ.get("BRAIN_EMBED_DEVICE") or _autodetect_device()
            model = SentenceTransformer(_MODEL_NAME, device=device)
            _EMBEDDER_CACHE = (model, device)
    return _EMBEDDER_CACHE


def _load_index(vectors_path: Path, meta_path: Path):
    """Load the .npy vectors and .jsonl metadata, cached by mtime.

    Cheap to call repeatedly; the actual disk read only fires when either
    index file changes (i.e. ingest rewrote it). Both mtimes are part of
    the cache key, so a search that lands between ingest writing the .npy
    and the .jsonl never serves a mismatched pair.
    """
    import numpy as np  # type: ignore[import-not-found]

    global _INDEX_CACHE
    try:
        key = (vectors_path.stat().st_mtime, meta_path.stat().st_mtime)
    except OSError:
        return None
    # Snapshot the global into a local before reading its fields: a
    # concurrent rebuild can reassign _INDEX_CACHE between the [1] and [2]
    # reads, pairing vectors from one generation with meta from another
    # (search's row-count guard catches only size, not equal-size content
    # mismatch).
    cache = _INDEX_CACHE
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    vectors = np.load(vectors_path)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = [json.loads(ln) for ln in fh if ln.strip()]
    with _CACHE_LOCK:
        _INDEX_CACHE = (key, vectors, meta)
    return vectors, meta


def _autodetect_device() -> str:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _chunks_for_record(rec: IndexRecord, paths: VaultPaths) -> list[Chunk]:
    """Chunk one record's processed markdown. Empty list when the file is
    missing or produces no embeddable chunks — shared by the full rebuild
    and the per-note upsert so both derive titles and rows identically."""
    if not rec.processed_path:
        return []
    path = paths.root / rec.processed_path
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []  # vanished/unreadable between stat and read: nothing to embed
    title = Path(rec.relative_path).stem.replace("_", " ").replace("-", " ").strip()
    # Curated knowledge notes (esp. one-line memory facts) are exempt from
    # the decorative-noise floor: a short fact must stay retrievable. Ingested
    # sources keep the floor to drop stray headings/boilerplate. This is
    # purely additive — notes with paragraphs above the floor are unchanged.
    min_chars = 1 if rec.extractor == KNOWLEDGE_EXTRACTOR else _MIN_CHARS
    return [
        Chunk(
            source_relative_path=rec.relative_path,
            source_hash=rec.source_hash,
            title=title,
            chunk_idx=i,
            text=chunk,
            origin=rec.extractor,
        )
        for i, chunk in enumerate(chunk_markdown(text, min_chars=min_chars))
    ]


def _meta_row(c: Chunk) -> dict[str, object]:
    """The on-disk JSONL row for one chunk. Single definition so upsert
    rows are byte-compatible with full-rebuild rows."""
    return {
        "source_relative_path": c.source_relative_path,
        "source_hash": c.source_hash,
        "title": c.title,
        "chunk_idx": c.chunk_idx,
        "text": c.text,
        "origin": c.origin,
        "model": _MODEL_NAME,
    }


def _write_index_files(
    vectors, rows: list[dict], *, vectors_path: Path, meta_path: Path
) -> None:
    """Atomic-ish writes, vectors FIRST then meta. np.save appends `.npy`
    if missing, so give it a path that already ends in `.npy` and rename
    to drop the `.tmp`. The mtime-keyed ``_load_index`` cache plus search's
    row-count check reject any reader that lands between the two renames.
    A crash BETWEEN the two os.replace calls can still leave a torn pair
    whose row counts coincidentally match (silent misalignment, not caught
    by the row-count guard); the recovery path is a full rebuild:
    ``uv run python scripts/ingest.py --rebuild-search-index``."""
    import numpy as np  # type: ignore[import-not-found]

    tmp_vec = vectors_path.with_name(vectors_path.stem + ".tmp.npy")
    np.save(tmp_vec, vectors)
    os.replace(tmp_vec, vectors_path)

    tmp_meta = meta_path.with_suffix(".jsonl.tmp")
    with tmp_meta.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_meta, meta_path)


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
        records = [
            r for r in candidates
            if r.status == "processed" and r.processed_path
        ]
        # Deliberate: only `processed` records are indexed. `partial` notes
        # (e.g. every PDF on a machine without MinerU — the pypdf fallback)
        # are NOT searchable. Surface the count loudly so a vault that seems
        # to "have nothing about X" is understood as an indexing-policy gap,
        # not an empty vault. To include them, re-extract with MinerU (or
        # change this filter).
        excluded_partial = sum(
            1 for r in candidates if r.status == "partial" and r.processed_path
        )
        if excluded_partial:
            logger.warning(
                "semantic: %d 'partial' record(s) are NOT indexed (not searchable) "
                "— install MinerU and re-ingest for full PDF extraction, or they "
                "stay text-only-but-invisible", excluded_partial,
            )
        if not records:
            logger.info("semantic: no processed records to index")
            return 0

        chunks = [c for rec in records for c in _chunks_for_record(rec, paths)]
        if not chunks:
            logger.info("semantic: 0 chunks after chunking")
            return 0

        logger.info(
            "semantic: encoding %d chunk(s) from %d source(s)",
            len(chunks), len(records),
        )

        try:
            import numpy as np  # type: ignore[import-not-found]
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

        texts = [c.text for c in chunks]
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
    encode: Callable[[list[str]], "np.ndarray"] | None = None,
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
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.warning("semantic: numpy not installed (%s) — skipping upsert", exc)
        return 0

    # The fallback rebuild happens AFTER the lock is released: build_index
    # takes the same (non-reentrant) lock, and the tiny unlock window is
    # harmless because a full rebuild reads index.jsonl + the notes, never
    # the files another writer might touch in between.
    needs_rebuild = False
    written = 0
    with _index_lock(paths):
        # Fresh read under the lock — deliberately not _load_index, whose
        # process-wide cache would serve (and then retain) pre-upsert rows.
        try:
            vectors = np.load(vectors_path)
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = [json.loads(ln) for ln in fh if ln.strip()]
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            logger.warning("semantic: failed to load index (%r) — full rebuild", exc)
            needs_rebuild = True
            meta, vectors = [], None

        if not needs_rebuild and len(meta) != vectors.shape[0]:
            logger.warning(
                "semantic: index size mismatch (vectors=%d, meta=%d) — full rebuild",
                vectors.shape[0], len(meta),
            )
            needs_rebuild = True

        # If the index was built by a DIFFERENT embedding model, patching new
        # rows in would silently mix incompatible vector spaces (the dim check
        # below only catches a dimension change, not a same-dim model swap).
        # The per-row ``model`` field exists precisely for this guard.
        if not needs_rebuild and any(
            m.get("model") not in (None, _MODEL_NAME) for m in meta
        ):
            logger.warning(
                "semantic: index has rows from a different model (want %s) — full rebuild",
                _MODEL_NAME,
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
                new_chunks.extend(_chunks_for_record(rec, paths))

            new_vecs = None
            if new_chunks:
                texts = [c.text for c in new_chunks]
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
                    if m.get("source_relative_path") not in updated
                ]
                kept_vecs = vectors[keep]
                rows = [meta[i] for i in keep]
                if new_chunks:
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
                logger.info(
                    "semantic: upsert wrote %d chunk(s) for %d note(s), %d row(s) total",
                    written, len(valid), len(rows),
                )

    if needs_rebuild:
        return build_index(paths, logger=logger)
    return written


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_VALID_MODES: tuple[str, ...] = ("dense", "lexical", "hybrid")
_RRF_K = 60          # reciprocal-rank-fusion damping (standard)
_CANDIDATES = 100    # per-ranking candidate pool feeding the fusion


def _hit_from_row(meta: list[dict], i: int, score: float) -> SearchHit:
    m = meta[i]
    return SearchHit(
        score=score,
        source_relative_path=m["source_relative_path"],
        title=m["title"],
        chunk_idx=int(m["chunk_idx"]),
        snippet=m["text"],
        origin=str(m.get("origin", "")),
    )


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
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        log.warning("semantic: numpy not installed (%s)", exc)
        return []

    loaded = _load_index(vectors_path, meta_path)
    if loaded is None:
        log.warning("semantic: index file disappeared between exists() check and load")
        return []
    vectors, meta = loaded
    if len(meta) != vectors.shape[0]:
        log.warning(
            "semantic: index size mismatch (vectors=%d, meta=%d) — rebuild required",
            vectors.shape[0], len(meta),
        )
        return []
    n = len(meta)
    cand = min(_CANDIDATES, n)

    # --- dense ranking (skipped for lexical mode -> no model load) ----------
    dense_rank: list[int] = []
    dense_scores = None
    if mode in ("dense", "hybrid"):
        try:
            model, _ = _load_embedder()
        except Exception as exc:  # noqa: BLE001
            if mode == "dense":
                log.warning("semantic: model load failed (%r)", exc)
                return []
            log.warning("semantic: model load failed (%r) — falling back to lexical", exc)
            mode = "lexical"
        if mode != "lexical":
            q_vec = np.asarray(
                model.encode([query], normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            dense_scores = (vectors @ q_vec.T).ravel()
            top = np.argpartition(-dense_scores, cand - 1)[:cand]
            dense_rank = [int(i) for i in top[np.argsort(-dense_scores[top])]]

    # --- lexical ranking ----------------------------------------------------
    lex_scores: dict[int, float] = {}
    lex_rank: list[int] = []
    if mode in ("lexical", "hybrid"):
        from . import lexical as _lex
        lidx = _lex.load_lexical_index(meta_path)
        if lidx is None:
            if mode == "lexical":
                log.warning("semantic: lexical index unavailable")
                return []
        else:
            lex_scores = _lex.score(lidx, query)
            lex_rank = [i for i, _s in sorted(
                lex_scores.items(), key=lambda kv: (-kv[1], kv[0]))][:cand]

    # --- fuse ---------------------------------------------------------------
    if mode == "dense":
        return [_hit_from_row(meta, i, float(dense_scores[i])) for i in dense_rank[:top_k]]
    if mode == "lexical":
        return [_hit_from_row(meta, i, float(lex_scores.get(i, 0.0))) for i in lex_rank[:top_k]]

    # hybrid: reciprocal-rank fusion of the two candidate lists.
    rrf: dict[int, float] = {}
    for rank, i in enumerate(dense_rank, start=1):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, i in enumerate(lex_rank, start=1):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (_RRF_K + rank)
    fused = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [_hit_from_row(meta, i, s) for i, s in fused]
