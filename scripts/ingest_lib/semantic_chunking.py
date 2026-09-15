"""Markdown chunking for ``ingest_lib.semantic``.

Split out of ``semantic.py`` (AUD-121) verbatim: the greedy paragraph-packing
logic that turns one processed note's markdown into ``Chunk`` objects, plus
the ``Chunk`` dataclass and ``_chunks_for_record`` that ties it to an
``IndexRecord``. None of this is monkeypatched by name in the tests, so it
carries no coupling back to ``semantic.py`` — see that module's docstring
for why other sections could not move as freely.

``semantic.py`` imports and re-exports every public and private name this
module defines that it used to define itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import VaultPaths
from .knowledge import KNOWLEDGE_EXTRACTOR
from .metadata import IndexRecord
from .notes import _split_frontmatter

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
    heading_path: str = ""      # e.g. "4 Graphs > 4.2 Directed" (for context)


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
