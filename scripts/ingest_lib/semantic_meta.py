"""Index row shapes and read-only lookups for ``ingest_lib.semantic``.

Split out of ``semantic.py`` (AUD-121) verbatim: the ``MetaRow``/``SearchHit``
shapes, the JSONL-row parser, the "is this record indexable" policy, the
per-chunk embed-text/model-tag derivation, and the read-only
``chunks_for_source`` helper. None of this is monkeypatched by name in the
tests, so it carries no coupling back to ``semantic.py`` — see that module's
docstring for why other sections could not move as freely. ``_MODEL_NAME``
comes from ``semantic_model`` (not from ``semantic.py`` itself, which a
sibling may not import) for the same reason.

``semantic.py`` imports and re-exports every public and private name this
module defines that it used to define itself.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypedDict

from .config import VaultPaths
from .metadata import IndexRecord
from .semantic_chunking import Chunk
from .semantic_model import _MODEL_NAME


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
