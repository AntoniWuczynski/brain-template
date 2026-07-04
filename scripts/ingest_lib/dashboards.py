"""Deterministic entity dashboards under ``knowledge/index/entities/``.

One Markdown table per entity group — people, organisations, projects,
meetings — rebuilt from :func:`relations.entity_notes`. Each dashboard
mirrors the concept-note shape exactly: managed frontmatter, an
``AUTO-GENERATED`` zone with the table, and a preserved user tail below
the end marker.

Dashboards live under ``knowledge/index/``, which sits OUTSIDE the
enrichment scan (``knowledge.KNOWLEDGE_NOTE_DIRS`` deliberately excludes
it), so they never feed back into semantic search, concept notes, or the
connection graph — they are pure views over the entity notes.

Dashboard layout::

    ---
    title: People
    type: dashboard
    count: N
    updated: '<ISO 8601 UTC>'
    ---

    <!-- AUTO-GENERATED-START -->

    # People

    > _one-line disclaimer_

    | Note | Current relations | Updated |
    | --- | --- | --- |
    | [[knowledge/people/anna]] | works_at -> [[knowledge/organisations/acme]] | ... |

    <!-- AUTO-GENERATED-END -->

    # Notes

    _(Your hand-written notes go here; preserved across re-runs.)_
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .concepts import (  # private helpers, but module-internal
    _AUTO_END,
    _AUTO_START,
    _existing_body_after_frontmatter,
    _existing_frontmatter,
)
from .config import VaultPaths
from .notes import _atomic_write, _split_frontmatter, fm_list, fm_scalar
from .relations import EntityInfo, entity_notes, normalize_target

# knowledge/ subdir -> dashboard title. Order is the render order; the
# other hand-edited areas (notes, research, university, assistant) are
# free-form and don't dashboard into a meaningful table.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("people", "People"),
    ("organisations", "Organisations"),
    ("projects", "Projects"),
    ("meetings", "Meetings"),
)

_MANAGED_KEYS: frozenset[str] = frozenset({"title", "type", "count", "updated"})


@dataclass(frozen=True)
class DashboardStats:
    written: int = 0
    unchanged: int = 0  # dashboards whose rendered content was already on disk
    # Vault-relative paths actually (re)written this run — excludes
    # unchanged notes, mirroring ConceptStats.written_paths.
    written_paths: tuple[str, ...] = ()


def rebuild_dashboards(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
) -> DashboardStats:
    """Group entity notes by knowledge subdir, write one dashboard per
    non-empty group. Skip-unchanged: a rebuild over an unchanged vault
    rewrites nothing (and re-timestamps nothing)."""
    paths.ensure()
    out_dir = paths.knowledge_index / "entities"

    by_group: dict[str, list[EntityInfo]] = {}
    # entity_notes is sorted by node id, so each group list is too.
    for info in entity_notes(paths).values():
        by_group.setdefault(info.node_id.split("/", 1)[0], []).append(info)

    written = 0
    unchanged = 0
    written_paths: list[str] = []
    for sub, title in _GROUPS:
        infos = by_group.get(sub, [])
        if not infos:
            # Empty group: write nothing. An existing dashboard is left
            # alone — deleting a file (and the user tail in it) because a
            # group emptied out would be destructive.
            continue
        target = out_dir / f"{sub}.md"
        header, rows = _rows_for_group(sub, infos, paths)
        try:
            wrote = _write_dashboard(
                target=target, title=title, header=header, rows=rows
            )
        except OSError as exc:
            logger.warning("dashboard '%s': failed to write (%s)", sub, exc)
            continue
        if wrote:
            written += 1
            written_paths.append(_vault_relative(target, paths))
        else:
            unchanged += 1

    stats = DashboardStats(
        written=written,
        unchanged=unchanged,
        written_paths=tuple(sorted(written_paths)),
    )
    logger.info("dashboards: written=%d unchanged=%d", stats.written, stats.unchanged)
    return stats


# ---------------------------------------------------------------------------
# row rendering
# ---------------------------------------------------------------------------

def _cell(text: str) -> str:
    """Markdown-table-safe cell text: a stray pipe or newline would break
    the row, so collapse whitespace and escape pipes."""
    return " ".join(text.split()).replace("|", "\\|")


def _fm_str(raw: object) -> str:
    """Frontmatter value -> stripped string. Unquoted YAML dates parse as
    date objects; their ISO form is what we want anyway."""
    if raw is None:
        return ""
    if isinstance(raw, (datetime, date)):
        return raw.isoformat()
    return str(raw).strip()


def _fm_list(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _note_frontmatter(path: Path) -> dict[str, Any]:
    """Tolerant frontmatter read for the columns EntityInfo doesn't carry
    (project status/topics, meeting date/attendees/project)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    frontmatter, _body = _split_frontmatter(text)
    return frontmatter


def _rows_for_group(
    sub: str, infos: list[EntityInfo], paths: VaultPaths
) -> tuple[tuple[str, ...], list[str]]:
    """Return ``(header_lines, row_lines)`` for one dashboard table."""
    if sub == "projects":
        header = ("| Note | Status | Topics | Updated |", "| --- | --- | --- | --- |")
        rows = []
        for info in infos:
            fm = _note_frontmatter(paths.root / info.rel_path)
            status = _cell(_fm_str(fm.get("status"))) or "—"
            topics = _cell(", ".join(_fm_list(fm.get("topics")))) or "—"
            updated = _cell(info.updated) or "—"
            rows.append(
                f"| [[knowledge/{info.node_id}]] | {status} | {topics} | {updated} |"
            )
        return header, rows

    if sub == "meetings":
        header = ("| Note | Date | Attendees | Project |", "| --- | --- | --- | --- |")
        entries: list[tuple[str, str, str]] = []
        for info in infos:
            fm = _note_frontmatter(paths.root / info.rel_path)
            date_s = _fm_str(fm.get("date"))
            attendees = "; ".join(
                f"[[knowledge/{normalize_target(a)}]]"
                for a in _fm_list(fm.get("attendees"))
            ) or "—"
            project = _fm_str(fm.get("project"))
            project_cell = f"[[knowledge/{normalize_target(project)}]]" if project else "—"
            row = (
                f"| [[knowledge/{info.node_id}]] | {_cell(date_s) or '—'} "
                f"| {attendees} | {project_cell} |"
            )
            entries.append((date_s, info.node_id, row))
        # Most recent meetings first is the useful order. Two stable sorts:
        # node id breaks date ties, and dateless meetings ("" sorts last
        # under reverse=True) sink to the bottom.
        entries.sort(key=lambda e: e[1])
        entries.sort(key=lambda e: e[0], reverse=True)
        return header, [e[2] for e in entries]

    # people / organisations: open relations only — '; '-joined,
    # sorted by (rel, target) so declaration-order edits don't churn rows.
    header = ("| Note | Current relations | Updated |", "| --- | --- | --- |")
    rows = []
    for info in infos:
        open_rels = sorted(
            (r for r in info.relations if not r.valid_until),
            key=lambda r: (r.rel, r.target),
        )
        rels = "; ".join(
            f"{r.rel} -> [[knowledge/{r.target}]]" for r in open_rels
        ) or "—"
        rows.append(f"| [[knowledge/{info.node_id}]] | {rels} | {_cell(info.updated) or '—'} |")
    return header, rows


# ---------------------------------------------------------------------------
# note writing (mirrors concepts._write_concept_note)
# ---------------------------------------------------------------------------

def _vault_relative(target: Path, paths: VaultPaths) -> str:
    """Vault-relative posix path for stats/commit lists. Falls back to the
    canonical location when ``knowledge`` sits outside ``root``
    (hand-built VaultPaths in tests can do that)."""
    try:
        return target.relative_to(paths.root).as_posix()
    except ValueError:
        return f"knowledge/index/entities/{target.name}"


def _write_dashboard(
    *,
    target: Path,
    title: str,
    header: tuple[str, ...],
    rows: list[str],
) -> bool:
    """Render and write one dashboard. Returns ``True`` if the file was
    written, ``False`` if skipped because nothing but the ``updated:``
    timestamp would change — same skip-unchanged contract as concept
    notes (``updated:`` means "content last changed", not "pipeline last
    ran")."""
    # Preserve user content below AUTO-GENERATED-END if the file exists.
    user_tail = ""
    existing = ""
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        end_pos = existing.find(_AUTO_END)
        if end_pos >= 0:
            user_tail = existing[end_pos + len(_AUTO_END) :].lstrip("\n")
        # A file without our markers was pre-created by hand: treat its
        # whole body as user content and prepend the generated block.
        elif _AUTO_START not in existing:
            user_tail = _existing_body_after_frontmatter(existing)

    if not user_tail.strip():
        user_tail = (
            "# Notes\n\n"
            "_(Your hand-written notes about these entities go here. "
            "Preserved across re-runs.)_\n"
        )

    # Pass through any extra user-added frontmatter keys we don't manage,
    # serialized YAML-safely — repr() mangled dates (datetime.date(...)) and
    # could emit invalid YAML that the next rebuild then silently dropped.
    user_fm = _existing_frontmatter(existing) if existing else {}
    extra_lines: list[str] = []
    for k, v in user_fm.items():
        if k in _MANAGED_KEYS:
            continue
        rendered = fm_list(v) if isinstance(v, list) else fm_scalar(v)
        extra_lines.append(f"{k}: {rendered}")

    def _frontmatter_lines(updated_value: str) -> list[str]:
        return [
            f"title: {fm_scalar(title)}",
            "type: dashboard",
            f"count: {len(rows)}",
            f"updated: '{updated_value}'",
            *extra_lines,
        ]

    body = (
        f"{_AUTO_START}\n\n"
        f"# {title}\n\n"
        f"> _Auto-generated dashboard — `knowledge/index/` is outside the "
        f"enrichment scan, so this table never feeds back into search or "
        f"concept notes. Edit anything below the **AUTO-GENERATED-END** "
        f"marker — those edits survive regeneration._\n\n"
        + "\n".join((*header, *rows))
        + "\n\n"
        + f"{_AUTO_END}\n\n"
        f"{user_tail.rstrip()}\n"
    )

    def _render(updated_value: str) -> str:
        return "---\n" + "\n".join(_frontmatter_lines(updated_value)) + "\n---\n\n" + body

    # Skip-unchanged: re-render with the EXISTING timestamp first. If that
    # reproduces the on-disk file byte-for-byte, only ``updated:`` would
    # move — don't write.
    if existing:
        prev_updated = user_fm.get("updated")
        if isinstance(prev_updated, str) and prev_updated:
            if _render(prev_updated) == existing:
                return False

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write(target, _render(now))
    return True


__all__ = ["DashboardStats", "rebuild_dashboards"]
