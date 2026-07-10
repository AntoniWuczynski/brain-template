"""Processing Dashboard + Manual Review notes, and a partial-retry pass.

Two auto-generated notes under ``knowledge/index/`` (outside the enrichment
scan, so they never feed back into search or concepts), rendered in the same
managed-zone / skip-unchanged style as the entity dashboards:

- ``Processing Dashboard.md`` — pipeline state at a glance: counts by
  status / extractor / extension, per-folder document counts, and an Inbox
  hygiene section that SHA-256-hashes every ``inbox/`` file against
  ``metadata/index.jsonl`` and labels each "already ingested — safe to
  delete" vs "pending".
- ``Manual Review.md`` — one row per ``partial`` / ``manual_review`` record
  (and any file physically sitting in ``archive/failed/``) with the recorded
  extractor error verbatim and the exact retry command.
- ``Now.md`` — a landing view: recently added sources (newest first), an
  at-a-glance area breakdown, and a "needs attention" panel (review backlog,
  inbox pending, unconsolidated assistant facts) linking to the other two.

Everything is derived from ``index.jsonl`` + the filesystem — no log
parsing, no timestamps in the body — so a rebuild over unchanged state is a
byte-for-byte no-op (only the frontmatter ``updated:`` stamp would move, and
skip-unchanged suppresses even that).

``retry_partial`` re-runs extraction for ``partial`` records (e.g. to pick
up MinerU after it was installed); ``archive/processed`` is regenerable, so
this is contract-safe.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Literal

from .config import VaultPaths
from .hashing import sha256_of
from .metadata import IndexRecord, latest_records_by_path
from .notes import _atomic_write

_AUTO_START = "<!-- AUTO-GENERATED-START -->"
_AUTO_END = "<!-- AUTO-GENERATED-END -->"

_DASHBOARD_NAME = "Processing Dashboard.md"
_REVIEW_NAME = "Manual Review.md"
_NOW_NAME = "Now.md"

_NOW_RECENT_LIMIT = 15


@dataclass(frozen=True)
class StatusStats:
    dashboard_written: bool = False
    review_written: bool = False
    now_written: bool = False
    inbox_pending: int = 0
    inbox_ingested: int = 0
    needs_review: int = 0
    written_paths: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# rendering (pure)
# ---------------------------------------------------------------------------

def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return out


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _dashboard_body(paths: VaultPaths, records: list[IndexRecord]) -> tuple[str, dict]:
    by_status = Counter(r.status for r in records)
    by_extractor = Counter(r.extractor for r in records)
    by_ext = Counter(r.extension for r in records)

    # Per-folder counts (the containing directory of each source).
    folder_status: dict[str, Counter] = {}
    for r in records:
        folder = str(Path(r.relative_path).parent) or "."
        folder_status.setdefault(folder, Counter())[r.status] += 1

    lines: list[str] = ["## Totals", ""]
    statuses: tuple[Literal["processed", "partial", "manual_review"], ...] = (
        "processed", "partial", "manual_review",
    )
    lines += _table(
        ("status", "count"),
        [(s, str(by_status[s])) for s in statuses],
    )
    lines += ["", "## By extractor", ""]
    lines += _table(("extractor", "count"),
                    [(k, str(v)) for k, v in sorted(by_extractor.items())])
    lines += ["", "## By extension", ""]
    lines += _table(("extension", "count"),
                    [(k or "(none)", str(v)) for k, v in sorted(by_ext.items())])

    lines += ["", "## By folder", ""]
    folder_rows: list[tuple[str, ...]] = []
    for folder in sorted(folder_status):
        c = folder_status[folder]
        total = sum(c.values())
        folder_rows.append((
            folder, str(c.get("processed", 0)), str(c.get("partial", 0)),
            str(c.get("manual_review", 0)), str(total),
        ))
    lines += _table(("folder", "processed", "partial", "review", "total"), folder_rows)

    # Inbox hygiene: hash each inbox file, compare against the known set.
    # List PENDING files (and unreadable ones) in detail — those are the
    # actionable rows; the already-ingested files get a single count line
    # (so a 600-file inbox doesn't render 600 "safe to delete" rows).
    known_hashes = {r.source_hash for r in records}
    pending_rows: list[tuple[str, ...]] = []
    pending = ingested = 0
    total_inbox = 0
    if paths.inbox.is_dir():
        for f in sorted(paths.inbox.rglob("*")):
            if not f.is_file() or f.name == ".DS_Store" or f.name.startswith("._"):
                continue
            total_inbox += 1
            rel = f.relative_to(paths.inbox).as_posix()
            try:
                h = sha256_of(f)
            except OSError:
                pending_rows.append((rel, "unreadable"))
                continue
            if h in known_hashes:
                ingested += 1
            else:
                pending_rows.append((rel, "pending"))
                pending += 1
    lines += ["", f"## Inbox ({pending} pending, {ingested} already ingested)", ""]
    if total_inbox == 0:
        lines += ["_(inbox is empty)_"]
    else:
        if pending_rows:
            lines += _table(("file", "state"), pending_rows)
        else:
            lines += ["_(nothing pending — every inbox file is already ingested)_"]
        if ingested:
            lines += [
                "",
                f"_{ingested} inbox file(s) match an ingested hash — safe to delete._",
            ]

    counts = {"inbox_pending": pending, "inbox_ingested": ingested}
    return "\n".join(lines), counts


def _review_body(paths: VaultPaths, records: list[IndexRecord]) -> tuple[str, int]:
    flagged = [r for r in records if r.status in ("partial", "manual_review")]
    flagged.sort(key=lambda r: (r.status, r.relative_path))

    lines: list[str] = [f"## Records needing attention ({len(flagged)})", ""]
    if flagged:
        rows: list[tuple[str, ...]] = []
        for r in flagged:
            retry = _retry_command(r)
            rows.append((r.relative_path, r.status, r.extractor,
                         (r.error or "").strip() or "—", retry))
        lines += _table(
            ("source", "status", "extractor", "error", "retry"), rows
        )
    else:
        lines += ["_(nothing needs review — every source extracted cleanly.)_"]

    # Files physically in archive/failed/ (belt-and-braces: a moved file with
    # no matching record still deserves a mention).
    failed_files: list[str] = []
    if paths.archive_failed.is_dir():
        recorded = {r.raw_path for r in flagged}
        for f in sorted(paths.archive_failed.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(paths.root).as_posix()
            if rel not in recorded:
                failed_files.append(rel)
    lines += ["", f"## Files in archive/failed/ ({len(failed_files)})", ""]
    if failed_files:
        lines += _table(("file",), [(p,) for p in failed_files])
    else:
        lines += ["_(none)_"]

    return "\n".join(lines), len(flagged)


def _unconsolidated_fact_count(paths: VaultPaths) -> int:
    """How many assistant facts are waiting to be consolidated — the ``.md``
    notes sitting in ``knowledge/assistant/inbox/``."""
    inbox = paths.knowledge / "assistant" / "inbox"
    if not inbox.is_dir():
        return 0
    return sum(1 for f in inbox.glob("*.md") if f.is_file())


def _now_body(paths: VaultPaths, records: list[IndexRecord]) -> tuple[str, dict]:
    """A landing view: what's recent and what needs attention. Derived from
    ``index.jsonl`` + the filesystem (``created_at`` is a stable record field),
    so a rebuild over unchanged state is a byte-for-byte no-op."""
    # Recently added sources: newest created_at first, path as the tie-break
    # so equal timestamps order deterministically.
    recent = sorted(records, key=lambda r: (r.created_at, r.relative_path), reverse=True)
    recent = recent[:_NOW_RECENT_LIMIT]

    needs_review = sum(1 for r in records if r.status in ("partial", "manual_review"))
    unconsolidated = _unconsolidated_fact_count(paths)

    # Inbox pending (files not yet matching an ingested hash).
    known_hashes = {r.source_hash for r in records}
    pending = 0
    if paths.inbox.is_dir():
        for f in paths.inbox.rglob("*"):
            if not f.is_file() or f.name == ".DS_Store" or f.name.startswith("._"):
                continue
            try:
                if sha256_of(f) not in known_hashes:
                    pending += 1
            except OSError:
                pending += 1

    lines: list[str] = ["## Needs attention", ""]
    lines += _table(
        ("item", "count", "where"),
        [
            ("Sources needing review", str(needs_review), "[[Manual Review]]"),
            ("Inbox files pending ingestion", str(pending), "[[Processing Dashboard]]"),
            ("Assistant facts unconsolidated", str(unconsolidated),
             "`knowledge/assistant/inbox/`"),
        ],
    )

    lines += ["", f"## Recently added sources ({len(recent)})", ""]
    if recent:
        rows: list[tuple[str, ...]] = [
            (r.created_at[:10] or "—", r.relative_path, str(r.status))
            for r in recent
        ]
        lines += _table(("added", "source", "status"), rows)
    else:
        lines += ["_(no sources ingested yet — drop files in `inbox/` and run ingest.)_"]

    # At a glance: total + by top-level folder.
    by_top: Counter = Counter()
    for r in records:
        top = Path(r.relative_path).parts[0] if Path(r.relative_path).parts else "."
        by_top[top] += 1
    lines += ["", f"## At a glance ({len(records)} sources)", ""]
    lines += _table(
        ("area", "sources"),
        [(k, str(v)) for k, v in sorted(by_top.items())],
    )

    return "\n".join(lines), {"needs_review": needs_review, "inbox_pending": pending}


def _retry_command(r: IndexRecord) -> str:
    if r.status == "partial":
        # The raw file is still in archive/raw; re-ingest re-extracts it
        # (partial records are not idempotency-skipped).
        return f"`uv run python scripts/ingest.py --path {r.raw_path}`"
    # manual_review: the file was moved to archive/failed; move it back to
    # inbox/ (fixing whatever failed) and re-ingest.
    return "move back to `inbox/` and re-ingest"


# ---------------------------------------------------------------------------
# managed-note write (skip-unchanged, preserves a user tail)
# ---------------------------------------------------------------------------

def _write_managed(target: Path, *, title: str, auto_body: str) -> bool:
    """Write a managed note (auto zone + preserved user tail). Returns True
    if written, False if only the ``updated:`` stamp would have changed."""
    existing = ""
    user_tail = ""
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        end = existing.find(_AUTO_END)
        if end >= 0:
            user_tail = existing[end + len(_AUTO_END):].lstrip("\n")
    if not user_tail.strip():
        user_tail = (
            "# Notes\n\n"
            "_(Your hand-written notes go here. Preserved across re-runs.)_\n"
        )

    def _render(updated: str) -> str:
        return (
            "---\n"
            f"title: {title}\n"
            "type: dashboard\n"
            f"updated: '{updated}'\n"
            "---\n\n"
            f"{_AUTO_START}\n\n"
            f"# {title}\n\n"
            "> _Auto-generated from `metadata/index.jsonl` + the filesystem. "
            "`knowledge/index/` is outside the enrichment scan. Edit below the "
            "**AUTO-GENERATED-END** marker — that survives regeneration._\n\n"
            f"{auto_body}\n\n"
            f"{_AUTO_END}\n\n"
            f"{user_tail.rstrip()}\n"
        )

    if existing:
        # Reuse the on-disk updated: stamp; if content is otherwise identical,
        # skip so a rebuild over unchanged state is commit-clean.
        prev = ""
        for line in existing.splitlines():
            if line.startswith("updated:"):
                prev = line.split(":", 1)[1].strip().strip("'\"")
                break
        if prev and _render(prev) == existing:
            return False

    now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write(target, _render(now))
    return True


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def rebuild_status(paths: VaultPaths, *, logger: logging.Logger) -> StatusStats:
    """Regenerate the Processing Dashboard + Manual Review notes."""
    paths.ensure()
    records = list(latest_records_by_path(paths.metadata_index_jsonl).values())

    dash_body, counts = _dashboard_body(paths, records)
    review_body, needs = _review_body(paths, records)
    now_body, _now_counts = _now_body(paths, records)

    dash_target = paths.knowledge_index / _DASHBOARD_NAME
    review_target = paths.knowledge_index / _REVIEW_NAME
    now_target = paths.knowledge_index / _NOW_NAME

    written_paths: list[str] = []
    dash_written = _write_managed(dash_target, title="Processing Dashboard", auto_body=dash_body)
    if dash_written:
        written_paths.append(dash_target.relative_to(paths.root).as_posix())
    review_written = _write_managed(review_target, title="Manual Review", auto_body=review_body)
    if review_written:
        written_paths.append(review_target.relative_to(paths.root).as_posix())
    now_written = _write_managed(now_target, title="Now", auto_body=now_body)
    if now_written:
        written_paths.append(now_target.relative_to(paths.root).as_posix())

    logger.info(
        "status: dashboard %s, review %s, now %s (%d inbox pending, %d need review)",
        "written" if dash_written else "unchanged",
        "written" if review_written else "unchanged",
        "written" if now_written else "unchanged",
        counts["inbox_pending"], needs,
    )
    return StatusStats(
        dashboard_written=dash_written,
        review_written=review_written,
        now_written=now_written,
        inbox_pending=counts["inbox_pending"],
        inbox_ingested=counts["inbox_ingested"],
        needs_review=needs,
        written_paths=tuple(written_paths),
    )


def retry_partial(paths: VaultPaths, *, logger: logging.Logger, dry_run: bool):
    """Re-run extraction for every ``partial`` record (e.g. after installing
    MinerU). archive/processed is regenerable, so this is contract-safe."""
    from .pipeline import plan_ingest, run_ingest  # local: avoid import cycle

    records = list(latest_records_by_path(paths.metadata_index_jsonl).values())
    targets = [
        paths.root / r.raw_path
        for r in records
        if r.status == "partial" and (paths.root / r.raw_path).is_file()
    ]
    logger.info("retry-partial: %d partial record(s) with a raw file to re-extract", len(targets))
    if not targets:
        from .pipeline import IngestStats
        return IngestStats()
    plan = plan_ingest(paths, sources=targets, from_archive=True, logger=logger)
    return run_ingest(paths, plan, dry_run=dry_run, logger=logger)
