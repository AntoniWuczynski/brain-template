"""Recency- and status-aware re-ranking on top of the semantic index.

``semantic.search`` ranks purely by cosine similarity — right for archival
material, wrong for *memory*: a meeting note from last week should outrank
a near-identical one from last year, and a note the MCP server stamped
``memory_status: superseded`` should sink no matter how well it matches.
This module re-ranks the existing index at query time. No new storage,
no rebuild step, nothing to drift out of sync:

    score = cosine * recency_weight * status_weight

- ``recency_weight`` is half-life decay on the hit's ``updated``
  timestamp: ``0.5 ** (age_days / halflife_days)``.
- ``status_weight`` reads ``memory_status`` frontmatter. Superseded notes
  are weighted down, not removed — provenance still matters, they just
  only surface when nothing fresher competes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Final
from collections.abc import Sequence

from . import semantic
from .config import VaultPaths
from .knowledge import KNOWLEDGE_EXTRACTOR
from .metadata import latest_records_by_path
from .notes import _split_frontmatter

# memory_status -> multiplicative weight. Anything not listed (including
# "", "unconsolidated", "consolidated", and ingested sources which have
# no memory_status at all) weighs 1.0: only an explicit "superseded"
# stamp demotes a note.
STATUS_WEIGHTS: Final[dict[str, float]] = {"superseded": 0.2}

# Valid tokens for the ``types`` filter: the hand-edited knowledge
# subdirs (a hit matches token t when its path starts with
# "knowledge/<t>/") plus the literal "archive" for ingested sources.
_TYPE_TOKENS: Final[tuple[str, ...]] = (
    "people",
    "organisations",
    "projects",
    "meetings",
    "notes",
    "research",
    "university",
    "assistant",
    "archive",
)

# Re-ranking reorders, so the final top_k can contain hits that cosine
# alone would have ranked far lower. Over-fetch candidates before
# re-ranking; the floor of 50 keeps small top_k requests from starving.
_CANDIDATE_FLOOR: Final[int] = 50
_CANDIDATE_MULTIPLIER: Final[int] = 5

# When a ``types`` filter is active the filter runs AFTER candidate fetch,
# so a sparse type (a few knowledge/people notes in a 1600-chunk index) can
# be starved out of a small candidate pool. Over-fetch a much larger pool
# before filtering. semantic.search widens its candidate pool to the
# requested top_k (min(n, max(100, top_k))) and the matmul scans every vector
# regardless, so a generous number is cheap and is actually honoured.
_FILTERED_CANDIDATE_CAP: Final[int] = 500


@dataclass(frozen=True)
class MemoryHit:
    """One re-ranked search result. ``score`` is the combined ranking
    score; the three factors are kept so callers can explain a ranking."""

    score: float            # cosine * recency * status_weight
    cosine: float           # raw similarity from semantic.search
    recency: float          # half-life decay factor, in (0, 1]
    status_weight: float    # 1.0 unless memory_status is down-weighted
    source_relative_path: str
    title: str
    snippet: str
    origin: str             # extractor tag ("knowledge-note" for vault notes)
    updated: str            # the raw timestamp string the decay was computed from
    chunk_idx: int


def _parse_when(raw: str) -> datetime | None:
    """Tolerant timestamp parse: ISO 8601 UTC ("2026-06-09T09:50:42Z",
    "+00:00" offsets) and bare YYYY-MM-DD dates. None when unparseable —
    callers treat that as 'timeless', never as an error."""
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)  # 3.11+: accepts the Z suffix
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Frontmatter dates are written in UTC by convention; a bare date
        # parses naive, so pin it rather than crash on aware-naive math.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def recency_weight(updated_iso: str, *, halflife_days: float, now: datetime) -> float:
    """Half-life decay on a note's ``updated`` timestamp.

    Empty or unparseable dates weigh 1.0: timeless notes (evergreen
    people/org/reference material with no meaningful clock) are not
    penalised for lacking one. Future dates also weigh 1.0 — clock skew
    or a forward-dated note should never *boost* past 1.0 nor crash.
    """
    parsed = _parse_when(updated_iso)
    if parsed is None:
        return 1.0
    age_days = (now - parsed).total_seconds() / 86400.0
    if age_days <= 0.0:
        return 1.0
    return 0.5 ** (age_days / halflife_days)


def _coerce_when(raw: object) -> str:
    """Frontmatter dates arrive as str when quoted, but yaml parses
    unquoted ``2026-06-09`` to a date object — normalise both to str."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (datetime, date)):
        return raw.isoformat()
    return ""


def _note_meta(root: Path, rel: str) -> tuple[str, str]:
    """(updated, memory_status) from a knowledge note's frontmatter.
    Missing/unreadable notes yield ("", "") — the hit still ranks, just
    without decay or demotion."""
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", ""
    frontmatter, _ = _split_frontmatter(text)
    updated = _coerce_when(frontmatter.get("updated"))
    raw_status = frontmatter.get("memory_status")
    status = raw_status.strip() if isinstance(raw_status, str) else ""
    return updated, status


def _passes_filter(rel: str, origin: str, allowed: frozenset[str]) -> bool:
    """Classify a hit by its ``origin``, not by a path PREFIX. A file
    dropped at ``inbox/knowledge/x.pdf`` is labelled ``knowledge/x.pdf`` but
    is an ingested SOURCE, not a vault note — only the ``origin`` tag
    (``KNOWLEDGE_EXTRACTOR`` for real notes) disambiguates the two."""
    if origin == KNOWLEDGE_EXTRACTOR:
        # A real vault note: classify by its knowledge/<token>/ subdir.
        # Generated areas (knowledge/index/, knowledge/concepts/) match no
        # token and are filtered out whenever a filter is active.
        return any(
            rel.startswith(f"knowledge/{t}/") for t in allowed if t != "archive"
        )
    # Ingested source (even one whose label starts with "knowledge/" because
    # it was dropped under inbox/knowledge/); "archive" selects all of them.
    return "archive" in allowed


def memory_search(
    paths: VaultPaths,
    query: str,
    *,
    top_k: int = 10,
    halflife_days: float = 30.0,
    types: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
    now: datetime | None = None,
) -> list[MemoryHit]:
    """Semantic search re-ranked by recency and memory status.

    ``types`` filters hits to the given knowledge subdirs and/or
    "archive" (None = no filter); unknown tokens raise ValueError so a
    typo can't silently return everything. ``now`` is injectable for
    tests; defaulting to wall-clock is fine here because this is a read
    path, not a rebuild path — determinism rules constrain what we
    *write*, and ranking freshness against the real clock is the point.
    """
    if types is not None:
        unknown = sorted(set(types) - set(_TYPE_TOKENS))
        if unknown:
            raise ValueError(
                f"unknown type token(s) {unknown} — valid: {', '.join(_TYPE_TOKENS)}"
            )
    if halflife_days <= 0:
        # recency_weight would ZeroDivisionError at 0 and boost old notes
        # (or OverflowError) for negatives. Validate at the boundary, like
        # `types` — this is exported public API.
        raise ValueError("halflife_days must be > 0")
    when = now or datetime.now(UTC)
    log = logger or logging.getLogger(__name__)

    # Filtering happens after fetch, so a filtered query must over-fetch a
    # large pool or a sparse type ranked deep in the index comes back empty.
    if types is None:
        fetch_k = max(_CANDIDATE_FLOOR, top_k * _CANDIDATE_MULTIPLIER)
    else:
        fetch_k = max(_FILTERED_CANDIDATE_CAP, top_k * _CANDIDATE_MULTIPLIER)
    candidates = semantic.search(paths, query, top_k=fetch_k, logger=log)
    if types is not None:
        allowed = frozenset(types)
        candidates = [
            c for c in candidates
            if _passes_filter(c.source_relative_path, c.origin, allowed)
        ]
    if not candidates:
        return []

    # Each distinct note is read at most once per call; the index.jsonl
    # join is loaded lazily so knowledge-only result sets never touch it.
    note_cache: dict[str, tuple[str, str]] = {}
    ingested_updated: dict[str, str] | None = None

    hits: list[MemoryHit] = []
    for cand in candidates:
        rel = cand.source_relative_path
        if cand.origin == KNOWLEDGE_EXTRACTOR:
            if rel not in note_cache:
                note_cache[rel] = _note_meta(paths.root, rel)
            updated, status = note_cache[rel]
        else:
            if ingested_updated is None:
                ingested_updated = {
                    p: r.updated_at
                    for p, r in latest_records_by_path(paths.metadata_index_jsonl).items()
                }
            updated, status = ingested_updated.get(rel, ""), ""
        recency = recency_weight(updated, halflife_days=halflife_days, now=when)
        status_weight = STATUS_WEIGHTS.get(status, 1.0)
        hits.append(
            MemoryHit(
                score=cand.score * recency * status_weight,
                cosine=cand.score,
                recency=recency,
                status_weight=status_weight,
                source_relative_path=rel,
                title=cand.title,
                snippet=cand.snippet,
                origin=cand.origin,
                updated=updated,
                chunk_idx=cand.chunk_idx,
            )
        )

    # Deterministic ordering: path + chunk_idx break score ties so equal
    # inputs always produce the same list.
    hits.sort(key=lambda h: (-h.score, h.source_relative_path, h.chunk_idx))
    return hits[:top_k]
