"""Treat curated ``knowledge/`` notes as first-class enrichment sources.

The enrichment pipelines (concept notes, semantic search, the connection
graph) historically consumed only ingested sources via
``metadata/index.jsonl``. Hand-written notes — project notes, research
notes, people/org notes — were invisible to all of them.

This module closes that gap without touching the index.jsonl contract:
``knowledge_records()`` scans the hand-edited knowledge areas and
synthesizes *virtual* :class:`IndexRecord` objects from each note's
frontmatter and body. Consumers simply merge these with the real records:

- ``relative_path`` / ``processed_path`` / ``index_note_path`` are all the
  note's vault-relative path. The note IS its own readable markdown, so
  semantic search chunks it directly and concept notes wikilink it as
  ``[[knowledge/...]]``.
- ``topics`` come from the note's frontmatter, exactly like summarizer
  topics on ingested sources — so concept notes and the co-occurrence
  graph pick them up with no further changes.
- Nothing is written: virtual records never land in ``index.jsonl``,
  which remains the record of *ingested* sources only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import VaultPaths
from .hashing import sha256_of
from .metadata import IndexRecord
from .notes import _split_frontmatter

# Hand-edited knowledge areas, mirroring the MCP server's write allowlist
# (mcp_server.config.WRITE_ALLOW_PREFIXES, minus inbox/) — keep in sync:
# the allowlist gains knowledge/meetings/ + knowledge/assistant/ in the
# same change-set that added them here.
# knowledge/index/ and knowledge/concepts/ are *generated from* sources and
# deliberately excluded: indexing them would double-count archive content
# and feed concept notes back into the concept graph.
KNOWLEDGE_NOTE_DIRS: tuple[str, ...] = (
    "assistant",
    "meetings",
    "notes",
    "organisations",
    "people",
    "projects",
    "research",
    "university",
)

# Extractor tag for virtual records. Also written into embeddings-meta
# rows as the hit's "origin" so the MCP search gate can distinguish a
# vault note from an ingested source whose label happens to start with
# "knowledge/" (e.g. a file dropped at inbox/knowledge/x.pdf).
KNOWLEDGE_EXTRACTOR = "knowledge-note"
_SUMMARY_MAX_CHARS = 300

# Promoted facts are moved under knowledge/assistant/archive/ as a historical
# record once consolidation folds their content into an entity note (or a
# digest). Re-indexing them would make the same fact double-retrievable —
# once from the entity-note Log line that now carries it, once from the
# archived original. So this subtree is excluded from the scan (and mirrored
# in semantic.upsert_notes). inbox/, digests/ and PROFILE.md stay scanned;
# KNOWLEDGE_NOTE_DIRS is deliberately left intact (other modules depend on
# its value) — the exclusion is applied inside the rglob walk instead.
_ASSISTANT_ARCHIVE_PREFIX = "knowledge/assistant/archive/"


@dataclass(frozen=True)
class KnowledgeScan:
    """Result of one scan: the synthesized records plus how many notes
    could not be read. A non-zero failure count means the record set is
    INCOMPLETE — destructive consumers (concept orphan-removal) must not
    treat missing notes as deliberately deleted."""

    records: list[IndexRecord] = field(default_factory=list)
    read_failures: int = 0


def knowledge_records(paths: VaultPaths) -> list[IndexRecord]:
    """Scan the hand-edited knowledge areas and synthesize one virtual
    record per Markdown note. Deterministic: sorted by relative path."""
    return scan_knowledge(paths).records


def scan_knowledge(
    paths: VaultPaths, *, logger: logging.Logger | None = None
) -> KnowledgeScan:
    """Like :func:`knowledge_records`, but reports read failures so callers
    can distinguish "note deleted" from "note unreadable right now"."""
    log = logger or logging.getLogger(__name__)
    records: list[IndexRecord] = []
    failures = 0
    for sub in KNOWLEDGE_NOTE_DIRS:
        base = paths.knowledge / sub
        if not base.is_dir():
            continue
        for md in sorted(base.rglob("*.md")):
            if not md.is_file():
                continue
            if md.relative_to(paths.root).as_posix().startswith(
                _ASSISTANT_ARCHIVE_PREFIX
            ):
                continue  # historical record; see _ASSISTANT_ARCHIVE_PREFIX
            try:
                rec = _record_for_note(md, paths)
            except FileNotFoundError:
                continue  # vanished between rglob and read: genuinely gone
            except OSError as exc:
                failures += 1
                log.warning("knowledge: failed to read %s (%s)", md, exc)
                continue
            if rec is not None:
                records.append(rec)
    records.sort(key=lambda r: r.relative_path)
    return KnowledgeScan(records=records, read_failures=failures)


def _record_for_note(md: Path, paths: VaultPaths) -> IndexRecord | None:
    text = md.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None

    frontmatter, body = _split_frontmatter(text)
    rel = md.relative_to(paths.root).as_posix()

    try:
        size = md.stat().st_size
    except OSError:
        size = len(text.encode())

    return IndexRecord(
        relative_path=rel,
        source_hash=sha256_of(md),
        size_bytes=size,
        extension=".md",
        extractor=KNOWLEDGE_EXTRACTOR,
        status="processed",
        raw_path=rel,
        processed_path=rel,
        index_note_path=rel,
        created_at=_str_or_empty(frontmatter.get("created")),
        updated_at=_str_or_empty(frontmatter.get("updated")),
        summary=_first_paragraph(body),
        topics=_normalise_topics(frontmatter.get("topics")),
    )


def _normalise_topics(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return []


def _str_or_empty(raw: object) -> str:
    return raw.strip() if isinstance(raw, str) else ""


def _first_paragraph(body: str) -> str:
    """First non-heading paragraph of the body, whitespace-collapsed —
    used as the snippet on concept-note source lines."""
    for block in body.split("\n\n"):
        lines = [
            ln for ln in block.strip().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not lines:
            continue
        collapsed = " ".join(" ".join(lines).split())
        if len(collapsed) > _SUMMARY_MAX_CHARS:
            collapsed = collapsed[: _SUMMARY_MAX_CHARS - 1] + "…"
        return collapsed
    return ""
