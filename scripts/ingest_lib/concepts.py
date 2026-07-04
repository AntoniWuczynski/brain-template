"""Build cross-source concept notes under ``knowledge/concepts/``.

For every topic — emitted by the summarizer on ingested sources, or
written in the frontmatter of curated ``knowledge/`` notes — write one
concept note that links every source and note in the vault that mentions
it. Anything the user hand-writes below the ``AUTO-GENERATED-END`` marker
is preserved on regeneration.

Concept-note layout::

    ---
    title: <Topic Name>
    type: concept
    sources_count: N
    updated: <ISO 8601 UTC>
    aliases: []
    ---

    <!-- AUTO-GENERATED-START -->

    # <Topic Name>

    > _Auto-generated index of every source in the vault that mentions
    > this concept. Edit anything below the AUTO-GENERATED-END marker —
    > those edits survive regeneration._

    ## Sources

    - [[knowledge/index/...]] — _<one-line summary>_
    - ...

    <!-- AUTO-GENERATED-END -->

    # Notes

    _(Your hand-written notes go here; preserved across re-runs.)_
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING

from .config import VaultPaths
from .knowledge import scan_knowledge
from .metadata import IndexRecord, latest_records_by_path
from .notes import (  # private helpers, but module-internal
    _atomic_write,
    _split_frontmatter,
    fm_list,
    fm_scalar,
)

if TYPE_CHECKING:
    # Type-only import keeps the concepts <- connections direction one-way at
    # runtime (connections imports slugify from here), avoiding a cycle.
    from .connections import Related

_AUTO_START = "<!-- AUTO-GENERATED-START -->"
_AUTO_END = "<!-- AUTO-GENERATED-END -->"

_NON_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ConceptStats:
    written: int = 0
    skipped: int = 0
    removed: int = 0    # concept notes whose sources are all gone
    unchanged: int = 0  # notes whose rendered content was already on disk
    # Vault-relative paths actually (re)written this run — a later stage
    # commits exactly these, so the set must exclude unchanged notes.
    written_paths: tuple[str, ...] = ()
    # Vault-relative paths of orphaned notes DELETED this run, so the same
    # later stage can stage the deletions too (git add stages removals).
    removed_paths: tuple[str, ...] = ()


def slugify(topic: str) -> str:
    """Filename-safe slug for a topic name.

    Two topic strings that slugify to the same value (e.g.
    ``Behaviour-Driven Development`` and ``behaviour-driven-development``)
    collapse to one concept note, so case and punctuation drift doesn't
    fragment the index.
    """
    return _NON_SLUG.sub("-", topic.strip().lower()).strip("-")


def rebuild_concepts(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
    related: dict[str, list[Related]] | None = None,
) -> ConceptStats:
    """Walk records, group by topic, write/refresh one concept note per topic.

    When ``related`` is supplied (slug -> ranked neighbours from the
    connection graph), each note also gets a "Related concepts" block inside
    its auto-generated zone. Returns counts of written/skipped/removed notes.
    """
    paths.ensure()
    concepts_dir = paths.knowledge / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    latest = latest_records_by_path(paths.metadata_index_jsonl)
    # Curated knowledge/ notes are first-class sources: their frontmatter
    # topics group into concepts exactly like summarizer topics do.
    scan = scan_knowledge(paths, logger=logger)
    all_records = list(latest.values()) + scan.records

    # Group by slug; preserve the most-common display variant per slug.
    # Dedupe (slug, record): hand-edited frontmatter can carry case/punct
    # variants of one topic ([RNG, rng]) and a source must count once.
    groups: dict[str, list[IndexRecord]] = defaultdict(list)
    display_votes: dict[str, Counter[str]] = defaultdict(Counter)
    grouped: set[tuple[str, str]] = set()
    for rec in all_records:
        for raw_topic in rec.topics or []:
            slug = slugify(raw_topic)
            if not slug:
                continue
            display_votes[slug][raw_topic] += 1
            if (slug, rec.relative_path) in grouped:
                continue
            grouped.add((slug, rec.relative_path))
            groups[slug].append(rec)

    valid_filenames: set[str] = set()
    written_count = 0
    unchanged_count = 0
    skipped_count = 0
    written_paths: list[str] = []

    for slug, records in groups.items():
        display = display_votes[slug].most_common(1)[0][0]
        target = concepts_dir / f"{slug}.md"
        valid_filenames.add(target.name)
        try:
            wrote = _write_concept_note(
                target=target,
                display_name=display,
                records=records,
                paths=paths,
                related_list=(related or {}).get(slug, []),
            )
        except OSError as exc:
            logger.warning("concept '%s': failed to write (%s)", display, exc)
            skipped_count += 1
            continue
        if wrote:
            written_count += 1
            written_paths.append(_vault_relative(target, paths))
        else:
            unchanged_count += 1

    # Drop concept notes whose topics are no longer referenced by any
    # record (e.g. the user removed a source). Only delete files we
    # generated ourselves: hand-written concept notes don't have the
    # AUTO-GENERATED markers and are left alone.
    #
    # Destructive pass — refuse it on an incomplete scan. If any knowledge
    # note failed to READ (vs being deleted), its concepts would look
    # orphaned and be unlinked, taking any hand-written tail below the
    # marker with them. Skip removal and let the next clean rebuild prune.
    if scan.read_failures:
        logger.warning(
            "concepts: %d knowledge note(s) unreadable — skipping orphan "
            "removal this run (would risk deleting live concept notes)",
            scan.read_failures,
        )
        stats = ConceptStats(
            written=written_count,
            skipped=skipped_count,
            removed=0,
            unchanged=unchanged_count,
            written_paths=tuple(sorted(written_paths)),
        )
        logger.info(
            "concepts: written=%d unchanged=%d skipped=%d removed=%d",
            stats.written, stats.unchanged, stats.skipped, stats.removed,
        )
        return stats
    removed = 0
    removed_paths: list[str] = []
    for existing in concepts_dir.glob("*.md"):
        if existing.name == ".gitkeep":
            continue
        if existing.name in valid_filenames:
            continue
        # Per-file guard, matching the write loop: one unreadable entry
        # (permission-denied, or a directory named *.md) must not abort the
        # whole rebuild after notes were already written.
        try:
            text = existing.read_text(encoding="utf-8", errors="replace")
            if _AUTO_START in text and _AUTO_END in text:
                existing.unlink()
                removed += 1
                removed_paths.append(_vault_relative(existing, paths))
                logger.info("concept: removed orphaned %s", existing.name)
        except OSError as exc:
            logger.warning("concept: could not process %s (%s) — skipping", existing.name, exc)
            continue
    stats = ConceptStats(
        written=written_count,
        skipped=skipped_count,
        removed=removed,
        unchanged=unchanged_count,
        written_paths=tuple(sorted(written_paths)),
        removed_paths=tuple(sorted(removed_paths)),
    )
    logger.info(
        "concepts: written=%d unchanged=%d skipped=%d removed=%d",
        stats.written, stats.unchanged, stats.skipped, stats.removed,
    )
    return stats


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _vault_relative(target: Path, paths: VaultPaths) -> str:
    """Vault-relative posix path for stats/commit lists. Falls back to the
    canonical concepts location when ``knowledge`` sits outside ``root``
    (hand-built VaultPaths in tests can do that)."""
    try:
        return target.relative_to(paths.root).as_posix()
    except ValueError:
        return f"knowledge/concepts/{target.name}"


def _related_reason(r: Related) -> str:
    """One-line, deterministic explanation of why two concepts are linked."""
    bits: list[str] = []
    if r.cooccurrence:
        n = int(r.cooccurrence)
        bits.append(f"co-occurs in {n} doc" + ("s" if n != 1 else ""))
    if r.semantic:
        bits.append(f"semantic {r.semantic:.2f}")
    return " · ".join(bits) if bits else "related"


def _write_concept_note(
    *,
    target: Path,
    display_name: str,
    records: list[IndexRecord],
    paths: VaultPaths,
    related_list: list[Related] | None = None,
) -> bool:
    """Render and write one concept note. Returns ``True`` if the file was
    written, ``False`` if skipped because nothing but the ``updated:``
    timestamp would change — that skip is what keeps a no-op rebuild from
    churning every concept note (and makes ``updated:`` mean "content last
    changed", not "pipeline last ran")."""
    # Preserve user content below AUTO-GENERATED-END if the file exists.
    user_tail = ""
    existing = ""
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        end_pos = existing.find(_AUTO_END)
        if end_pos >= 0:
            user_tail = existing[end_pos + len(_AUTO_END) :].lstrip("\n")
        # If the file exists without our markers, treat the entire body as
        # user content (i.e. the user pre-created this concept note by
        # hand). We then prepend the generated block above their content.
        elif _AUTO_START not in existing:
            user_tail = _existing_body_after_frontmatter(existing)

    if not user_tail.strip():
        user_tail = (
            "# Notes\n\n"
            "_(Your hand-written notes about this concept go here. "
            "Preserved across re-runs.)_\n"
        )

    # Sort sources by relative_path for stable, deterministic output.
    records_sorted = sorted(records, key=lambda r: r.relative_path)

    # Frontmatter — preserve user-added keys (e.g. aliases) if they were
    # in the existing file.
    user_fm = _existing_frontmatter(existing) if existing else {}
    aliases_str = fm_list(user_fm.get("aliases"))
    extra_lines: list[str] = []
    # Pass through any extra user-added keys we don't manage, serialized
    # YAML-safely (repr() mangled dates and could emit invalid YAML that the
    # next rebuild then silently dropped).
    for k, v in user_fm.items():
        if k in {"title", "type", "sources_count", "updated", "aliases"}:
            continue
        rendered = fm_list(v) if isinstance(v, list) else fm_scalar(v)
        extra_lines.append(f"{k}: {rendered}")

    def _frontmatter_lines(updated_value: str) -> list[str]:
        return [
            f"title: {fm_scalar(display_name)}",
            "type: concept",
            f"sources_count: {len(records_sorted)}",
            f"updated: '{updated_value}'",
            f"aliases: {aliases_str}",
            *extra_lines,
        ]

    sources_lines: list[str] = []
    for r in records_sorted:
        wikilink = _index_note_wikilink(r, paths)
        snippet = _short_snippet(r)
        sources_lines.append(f"- {wikilink}{snippet}")

    related_lines = [
        f"- [[knowledge/concepts/{rel.slug}]] — _{_related_reason(rel)}_"
        for rel in (related_list or [])
    ]
    related_block = (
        f"## Related concepts ({len(related_lines)})\n\n"
        + "\n".join(related_lines)
        + "\n\n"
        if related_lines
        else ""
    )

    body = (
        f"{_AUTO_START}\n\n"
        f"# {display_name}\n\n"
        f"> _Auto-generated index of every source in the vault that mentions "
        f"this concept. Edit anything below the **AUTO-GENERATED-END** marker "
        f"— those edits survive regeneration._\n\n"
        f"## Sources ({len(records_sorted)})\n\n"
        + "\n".join(sources_lines)
        + "\n\n"
        + related_block
        + f"{_AUTO_END}\n\n"
        f"{user_tail.rstrip()}\n"
    )

    def _render(updated_value: str) -> str:
        return "---\n" + "\n".join(_frontmatter_lines(updated_value)) + "\n---\n\n" + body

    # Skip-unchanged: re-render with the EXISTING timestamp first. If that
    # reproduces the on-disk file byte-for-byte, only ``updated:`` would
    # move — don't write. (No existing file, or no parseable timestamp:
    # always write.)
    if existing:
        prev_updated = user_fm.get("updated")
        if isinstance(prev_updated, str) and prev_updated:
            if _render(prev_updated) == existing:
                return False

    now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write(target, _render(now))
    return True


def _index_note_wikilink(rec: IndexRecord, paths: VaultPaths) -> str:
    """Return ``[[knowledge/index/<rel without .md>]]`` if available."""
    if rec.index_note_path:
        path = rec.index_note_path
        if path.endswith(".md"):
            path = path[:-3]
        return f"[[{path}]]"
    return f"`{rec.relative_path}`"


def _short_snippet(rec: IndexRecord) -> str:
    if rec.summary:
        # Truncate to a single line of reasonable length.
        s = rec.summary.replace("\n", " ").strip()
        if len(s) > 140:
            s = s[:137] + "…"
        return f" — _{s}_"
    return ""


def _existing_body_after_frontmatter(text: str) -> str:
    # Delegate to the canonical parser (CRLF-tolerant, refuses to strip a
    # leading '---' horizontal rule): a hand-created concept note in a CRLF
    # editor otherwise had its whole body — frontmatter included — treated as
    # user_tail and re-embedded below the marker.
    return _split_frontmatter(text)[1]


def _existing_frontmatter(text: str) -> dict[str, object]:
    return _split_frontmatter(text)[0]
