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
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict
from collections.abc import Callable, Iterator

if TYPE_CHECKING:
    # numpy is imported lazily inside functions (torch-free CLIs shouldn't pay
    # the import cost); this makes the "np.ndarray" annotations resolvable for
    # ruff and mypy without importing it at module load.
    import numpy as np
    from .lexical import LexicalIndex

from . import rank as rank_lib
from .config import RankingConfig, VaultPaths, ranking_config
from .knowledge import KNOWLEDGE_EXTRACTOR, _record_for_note, knowledge_records
from .metadata import IndexRecord, latest_records_by_path
from .notes import _split_frontmatter


_MODEL_NAME = "BAAI/bge-small-en-v1.5"
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
_TARGET_TOKENS = 400          # rough budget per chunk; English ≈ chars/4
_TARGET_CHARS = _TARGET_TOKENS * 4
_MIN_CHARS = 80               # skip tiny chunks (headings on their own line)

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


@dataclass(frozen=True)
class Chunk:
    """One chunk to be embedded."""
    source_relative_path: str   # the original raw path
    source_hash: str
    title: str
    chunk_idx: int              # 0-based, within the source
    text: str
    origin: str = ""            # the record's extractor (e.g. "knowledge-note")
    heading_path: str = ""      # e.g. "4 Graphs > 4.2 Directed" (for context)


class MetaRow(TypedDict):
    """One row of ``metadata/embeddings_meta.jsonl`` — the on-disk shape
    written by ``_meta_row``/``_write_index_files`` and read back by
    ``_load_index``/``chunks_for_source`` (narrowed once, on load, by
    ``_row_from_json``). ``origin``, ``heading_path`` and ``model`` were
    added to the schema after ``source_relative_path``/``source_hash``/
    ``title``/``chunk_idx``/``text``, so a row from an older index
    generation may lack them; ``_row_from_json`` defaults those three to
    ``""`` rather than failing, while a missing/wrong-typed core field is
    treated as corruption. ``gen`` is the same story: it identifies the
    generation of ``embeddings.npy`` these rows were written against and is
    stamped by ``_write_index_files``, not by ``_meta_row`` (which has no
    matrix to stamp from); rows from before it existed carry ``""``."""
    source_relative_path: str
    source_hash: str
    title: str
    chunk_idx: int
    text: str
    origin: str
    heading_path: str
    model: str
    gen: str


def _row_from_json(obj: object, index: int) -> MetaRow:
    """Narrow one decoded ``embeddings_meta.jsonl`` line to a ``MetaRow``.

    The file is written only by this module, so a missing/wrong-typed CORE
    field (``source_relative_path``, ``source_hash``, ``title``,
    ``chunk_idx``, ``text``) is corruption — raise loudly with the row
    index rather than silently coercing. ``origin``/``heading_path``/
    ``model`` default to ``""`` when absent (see ``MetaRow``).
    """
    if not isinstance(obj, dict):
        raise ValueError(f"embeddings_meta.jsonl row {index}: not a JSON object")
    for field in ("source_relative_path", "source_hash", "title", "text"):
        if not isinstance(obj.get(field), str):
            raise ValueError(
                f"embeddings_meta.jsonl row {index}: {field!r} is missing or not a string"
            )
    chunk_idx = obj.get("chunk_idx")
    if not isinstance(chunk_idx, int):
        raise ValueError(
            f"embeddings_meta.jsonl row {index}: 'chunk_idx' is missing or not an int"
        )
    origin = obj.get("origin")
    heading_path = obj.get("heading_path")
    model = obj.get("model")
    gen = obj.get("gen")
    return MetaRow(
        source_relative_path=obj["source_relative_path"],
        source_hash=obj["source_hash"],
        title=obj["title"],
        chunk_idx=chunk_idx,
        text=obj["text"],
        origin=origin if isinstance(origin, str) else "",
        heading_path=heading_path if isinstance(heading_path, str) else "",
        model=model if isinstance(model, str) else "",
        gen=gen if isinstance(gen, str) else "",
    )


@dataclass(frozen=True)
class SearchHit:
    """One search result.

    ``score`` is whatever the requested ``mode`` ranks by, and the three
    are NOT comparable — it used to be documented as "cosine similarity",
    which is true of only one of them:

    - ``dense``   — cosine similarity, in [-1, 1] (usually 0..1).
    - ``lexical`` — a BM25 score: unbounded above, 0 for no term overlap.
    - ``hybrid``  — a reciprocal-rank-fusion score (the default), bounded
      by ``2/(_RRF_K + 1)`` ≈ 0.0328. It ranks by POSITION, so it says
      nothing about match quality: the top hit for a nonsense query scores
      the same as the top hit for a perfect one. Do not present it as a
      similarity or threshold on it.
    """
    score: float
    source_relative_path: str
    title: str
    chunk_idx: int
    snippet: str                # the actual chunk text
    origin: str = ""            # "" on indexes built before origin existed
    degraded: bool = False      # True when a hybrid query fell back to
                                # lexical-only because the embedder failed,
                                # so a caller can tell BM25 ranks from a
                                # real fusion instead of guessing


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Anchored on the literal ``> Source:`` line that ``write_processed_note``
# always emits as the first blockquote line — NOT on any ``# heading … >
# quote … ---`` shape. A DOTALL ``.*?`` here used to span arbitrary body
# lines until a later blockquote + ``---``, silently eating a curated note's
# body that happened to precede a callout and a horizontal rule.
_HEADER_BLOCK_RE = re.compile(
    r"\A#\s[^\n]*\n+>\s*Source:[^\n]*\n(?:>\s[^\n]*\n)*\n+---\n+",
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
    body content the way a bare regex scan would. A leading UTF-8 BOM
    (notes written by some Windows editors) is dropped first: neither
    parser recognises ``﻿---`` as a fence, so without this the whole
    YAML block would be embedded as body text."""
    _, text = _split_frontmatter(text.lstrip("﻿"))
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


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


def _update_heading_stack(stack: list[tuple[int, str]], block: str) -> None:
    """If ``block`` is a lone ATX heading line, fold it into the running
    heading stack (`##` closes any `##`/deeper, then pushes)."""
    if "\n" in block:
        return
    m = _HEADING_LINE_RE.match(block.strip())
    if not m:
        return
    level = len(m.group(1))
    htext = m.group(2).strip()
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, htext))


def _is_lone_heading(block: str) -> bool:
    """Is this block nothing but a single ATX heading line?"""
    return "\n" not in block and _HEADING_LINE_RE.match(block.strip()) is not None


def _split_oversize_block(block: str) -> list[str]:
    """Cut a block wider than ``_TARGET_CHARS`` into budget-sized pieces.

    A block bigger than the budget used to be emitted whole, so the parts
    past the embedder's context window (512 tokens for bge-small) never
    reached the vector at all — extracted tables and bibliographies arrive
    as one blank-line-free block and can run to tens of thousands of
    characters. Cutting on line boundaries keeps those row-per-line
    structures intact; a single line longer than the budget (a minified
    blob, a wrapped-free paragraph) still has to be cut mid-line, because
    the alternative is content that cannot be embedded at all."""
    if len(block) <= _TARGET_CHARS:
        return [block]
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in block.splitlines():
        segments = [line[i:i + _TARGET_CHARS] for i in range(0, len(line), _TARGET_CHARS)] or [""]
        for seg in segments:
            if buf and buf_len + len(seg) + 1 > _TARGET_CHARS:
                pieces.append("\n".join(buf))
                buf, buf_len = [], 0
            buf.append(seg)
            buf_len += len(seg) + 1
    if buf:
        pieces.append("\n".join(buf))
    return pieces


def _pack_blocks_with_headings(text: str, min_chars: int) -> list[tuple[str, str]]:
    """Greedy-pack paragraphs into ~``_TARGET_CHARS`` chunks, tracking the
    heading path in force at each chunk's start. Returns ``(chunk_text,
    heading_path)`` pairs; the ``chunk_text`` is byte-identical to what
    ``chunk_markdown`` produced before heading tracking was added."""
    blocks = _split_into_blocks(_strip_processed_header(text))
    stack: list[tuple[int, str]] = []
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_len = 0
    buf_head = ""
    in_fence = False   # inside a ``` / ~~~ code fence: '# x' lines aren't headings
    for block in blocks:
        started_in_fence = in_fence
        for ln in block.splitlines():
            s = ln.lstrip()
            if s.startswith("```") or s.startswith("~~~"):
                in_fence = not in_fence
        # Only update the heading stack for a block that is wholly OUTSIDE any
        # code fence — a '# comment' inside fenced code is not a section head.
        if not started_in_fence and not in_fence:
            _update_heading_stack(stack, block)
        cur_head = " > ".join(h for _lvl, h in stack)
        block_len = len(block)
        # A buffer holding nothing but a heading line is not a chunk: emitting
        # it alone leaves a "## Log" row of pure noise AND strips the section
        # head off the paragraph that follows. Carry it into the next chunk.
        carry = buf if len(buf) == 1 and _is_lone_heading(buf[0]) else []
        if block_len > _TARGET_CHARS:
            # Block is bigger than the budget. Flush what we've got, then
            # cut the block on line boundaries so no piece overflows the
            # embedder's window (see _split_oversize_block).
            if buf and not carry:
                chunks.append(("\n\n".join(buf), buf_head))
            pieces = _split_oversize_block(block)
            chunks.append(("\n\n".join([*carry, pieces[0]]), buf_head if carry else cur_head))
            chunks.extend((p, cur_head) for p in pieces[1:])
            buf, buf_len, buf_head = [], 0, ""
        elif buf and buf_len + block_len + 2 > _TARGET_CHARS:
            if carry:
                buf = [*carry, block]
                buf_len += block_len + 2
            else:
                chunks.append(("\n\n".join(buf), buf_head))
                buf = [block]
                buf_len = block_len
                buf_head = cur_head
        else:
            if not buf:
                buf_head = cur_head
            buf.append(block)
            buf_len += block_len + 2
    if buf:
        chunks.append(("\n\n".join(buf), buf_head))
    return [(t, h) for t, h in chunks if len(t) >= min_chars]


def chunk_markdown(text: str, *, min_chars: int = _MIN_CHARS) -> list[str]:
    """Greedy-pack paragraphs into ~``_TARGET_CHARS`` chunks.

    ``min_chars`` is the floor below which a chunk is dropped as noise.
    Curated knowledge notes pass a lower floor (see ``_chunks_for_record``):
    a one-line memory-fact note is legitimately short, and dropping it would
    make the fact permanently unsearchable AND permanently flag it as
    index drift in the sweep."""
    return [t for t, _h in _pack_blocks_with_headings(text, min_chars)]


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


def _chunks_for_record(
    rec: IndexRecord, paths: VaultPaths, *, text: str | None = None
) -> list[Chunk]:
    """Chunk one record's processed markdown. Empty list when the file is
    missing or produces no embeddable chunks — shared by the full rebuild
    and the per-note upsert so both derive titles and rows identically.

    ``text`` lets a caller that has already read the file hand the exact
    bytes it hashed straight in, instead of paying a second read that a
    concurrent editor could make disagree with ``rec.source_hash``."""
    if not rec.processed_path:
        return []
    if text is None:
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
            heading_path=heading_path,
        )
        for i, (chunk, heading_path) in enumerate(
            _pack_blocks_with_headings(text, min_chars)
        )
    ]


def _meta_row(c: Chunk) -> MetaRow:
    """The on-disk JSONL row for one chunk. Single definition so upsert
    rows are byte-compatible with full-rebuild rows. ``gen`` is filled in
    by ``_write_index_files``, which is where the vector matrix — and so
    the generation being written — is known."""
    return {
        "source_relative_path": c.source_relative_path,
        "source_hash": c.source_hash,
        "title": c.title,
        "chunk_idx": c.chunk_idx,
        "text": c.text,
        "origin": c.origin,
        "heading_path": c.heading_path,
        "model": _embed_model_tag(),
        "gen": "",
    }


def _embed_model_tag() -> str:
    """The vector space a row belongs to, recorded per row.

    The model name alone does not identify it: ``BRAIN_EMBED_HEADING_CONTEXT``
    changes what gets embedded (see ``_embed_text``) without changing the
    model or the dimension, and the MCP server (launchd env) reads that
    variable independently of the CLI (shell env) — so an upsert from one
    could patch rows into a matrix built by the other. Folding the flag into
    this tag makes the existing model-mismatch guard in ``upsert_notes`` see
    the difference and force a full rebuild."""
    if os.environ.get("BRAIN_EMBED_HEADING_CONTEXT", "0") == "1":
        return f"{_MODEL_NAME}+heading"
    return _MODEL_NAME


def _embed_text(c: Chunk) -> str:
    """The string actually sent to the embedder for chunk ``c``.

    Default: the raw chunk text (the on-disk ``text`` / BM25 source). With
    ``BRAIN_EMBED_HEADING_CONTEXT=1`` the chunk's title + heading path is
    prepended so a lecture-slide paragraph embeds with its section context
    (e.g. "COMP0005 > 4.2 Directed Graphs\\n<text>"). This changes the
    passage vectors, so enabling it requires a full ``--rebuild-search-index``;
    measure the effect with ``scripts/eval_retrieval.py --compare-modes`` /
    ``--write-report`` before keeping it (see TODO.md / IDEAS.md)."""
    if os.environ.get("BRAIN_EMBED_HEADING_CONTEXT", "0") != "1":
        return c.text
    parts = [p for p in (c.title, c.heading_path) if p]
    ctx = " > ".join(parts)
    return f"{ctx}\n{c.text}" if ctx else c.text


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


# `partial` is not one thing. Two extractors downgrade a source to `partial`
# purely because it CONTAINS visuals they cannot read: docx (embedded
# drawings/images/objects) and pptx (pictures/charts) each set `partial` at
# exactly one site, and at that site the text came out whole — it is the
# pictures that carry no extractable text. Withholding those from search
# hides text that is complete and on disk (26 decks on this vault). Every
# other `partial` means the extracted TEXT is short of the source — a PDF
# that fell back to pypdf, a file cut at the size cap, a transcript whose
# cues would not parse, a meeting with no body — and those stay out: ranking
# half a document as if it were whole is worse than a miss.
#
# Keyed on the extractor rather than the note prose because the note strings
# are human-facing and free to be reworded. If either extractor grows a
# second `partial` site for a text-incomplete case, it must stop being
# listed here.
_VISUALS_ONLY_PARTIAL_EXTRACTORS: frozenset[str] = frozenset({"docx", "pptx"})


def _is_indexable(rec: IndexRecord) -> bool:
    """Whether this record's processed markdown belongs in the search index."""
    if not rec.processed_path:
        return False
    if rec.status == "processed":
        return True
    return (
        rec.status == "partial"
        and rec.extractor in _VISUALS_ONLY_PARTIAL_EXTRACTORS
    )


def _by_extension(records: list[IndexRecord]) -> str:
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.extension or "(none)"] = counts.get(rec.extension or "(none)", 0) + 1
    return ", ".join(f"{ext}={n}" for ext, n in sorted(counts.items()))


def _reextract_advice(records: list[IndexRecord]) -> str:
    """Advice fitting what was actually withheld. MinerU is the fix for PDFs
    and for nothing else, so only say so when a PDF is among them."""
    if any(rec.extension == ".pdf" for rec in records):
        return (
            "Install MinerU and re-ingest for full PDF extraction; re-extract "
            "the rest at their source."
        )
    return "Re-extract and re-ingest them to make their text searchable."


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


def _hit_from_row(
    meta: list[MetaRow], i: int, score: float, *, degraded: bool = False
) -> SearchHit:
    m = meta[i]
    return SearchHit(
        score=score,
        source_relative_path=m["source_relative_path"],
        title=m["title"],
        chunk_idx=m["chunk_idx"],
        snippet=m["text"],
        origin=m["origin"],
        degraded=degraded,
    )


def chunks_for_source(paths: VaultPaths, source_relative_path: str) -> list[MetaRow]:
    """Every indexed chunk of one source, ordered by ``chunk_idx``.

    A read-only view over ``embeddings_meta.jsonl`` (no vectors, no model) so
    a caller can expand a single search hit into its neighbouring chunks
    instead of reading the whole file. Empty list when the index is absent or
    the source has no chunks. Rows are keyed by ``(source_relative_path,
    chunk_idx)`` — chunk indices are relative to the current index generation
    and shift when the source is re-chunked (e.g. after a rebuild)."""
    meta_path = paths.metadata / "embeddings_meta.jsonl"
    if not meta_path.exists():
        return []
    try:
        with meta_path.open("r", encoding="utf-8", errors="replace") as fh:
            rows = [
                _row_from_json(json.loads(ln), i)
                for i, ln in enumerate(fh) if ln.strip()
            ]
    except (OSError, ValueError):
        return []
    selected = [r for r in rows if r["source_relative_path"] == source_relative_path]
    selected.sort(key=lambda r: r["chunk_idx"])
    return selected


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
