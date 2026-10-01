"""A source's table of contents, derived from its indexed chunks.

Consecutive chunks sharing a ``heading_path`` form one section. Each section
carries a short preview of its opening text, so sections under a repeated
generic heading (slide decks: "Demo", "Key idea") stay distinguishable.
Shared by the ``vault_outline`` MCP tool and ``scripts/eval_sections.py``.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .semantic_meta import MetaRow

PREVIEW_CHARS = 100
_WS = re.compile(r"\s+")


@dataclass
class Section:
    heading_path: str
    first_chunk: int
    last_chunk: int
    chars: int
    preview: str


def _preview(text: str) -> str:
    """Opening text of a chunk, minus ATX heading lines, whitespace-collapsed."""
    body = " ".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    return _WS.sub(" ", body).strip()[:PREVIEW_CHARS]


def build_outline(rows: Sequence[MetaRow]) -> list[Section]:
    """Group ``rows`` (one source's chunks, ordered by ``chunk_idx``) into
    consecutive runs sharing a heading path."""
    sections: list[Section] = []
    for r in rows:
        last = sections[-1] if sections else None
        if last is not None and last.heading_path == r["heading_path"]:
            last.last_chunk = r["chunk_idx"]
            last.chars += len(r["text"])
        else:
            sections.append(Section(
                heading_path=r["heading_path"], first_chunk=r["chunk_idx"],
                last_chunk=r["chunk_idx"], chars=len(r["text"]), preview=_preview(r["text"]),
            ))
    return sections
