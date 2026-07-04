#!/usr/bin/env python3
"""brain memory-consolidation CLI.

Examples:
    uv run python scripts/consolidate.py --dry-run
    uv run python scripts/consolidate.py
    uv run python scripts/consolidate.py --as-of 2026-06-12 --stale-days 14
    uv run python scripts/consolidate.py --min-confirmations 2 --no-reindex

The pass itself lives in ``scripts/ingest_lib/consolidate.py`` —
deterministic counters and thresholds, no LLM. This wrapper adds the
per-run log (AGENTS.md rule 5) and the post-run enrichment refresh
(semantic upsert + connection graph + concept notes), mirroring
``scripts/ingest.py``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly
# (i.e. without ``uv run`` having installed the package yet).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_lib import (  # noqa: E402
    default_paths,
    rebuild_concepts,
    rebuild_connections,
    rebuild_dashboards,
)
from ingest_lib.consolidate import ConsolidateStats, consolidate  # noqa: E402
from ingest_lib.logging_setup import _utc_converter  # noqa: E402  # module-internal reuse
from ingest_lib.semantic import upsert_notes  # noqa: E402


def _parse_as_of(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {raw!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-consolidate",
        description=(
            "Promote confirmed assistant memory facts from "
            "knowledge/assistant/inbox/ into entity notes, archive the "
            "originals, and digest stale leftovers. Deterministic; no LLM."
        ),
        epilog=(
            "Run this when the MCP server is idle. Consolidation takes no "
            "cross-process lock; it rewrites entity notes and unlinks inbox "
            "copies directly on disk, so a server writing the same note at "
            "the same instant can lose or duplicate a write. For a "
            "single-user vault, just avoid consolidating mid-conversation "
            "with the assistant."
        ),
    )
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Reference date for staleness, archive month and the "
            "consolidated: stamp (default: today, UTC)."
        ),
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=30,
        help="Unconsolidated facts older than this many days are digested (default: 30).",
    )
    parser.add_argument(
        "--min-confirmations",
        type=int,
        default=3,
        help="Promote unapproved facts at this confirmation count (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan but do not write anything. Still creates a log file.",
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help=(
            "Skip the post-run enrichment refresh (semantic upsert over "
            "touched/moved notes + connection graph + concept notes + "
            "entity dashboards)."
        ),
    )
    return parser


def _configure_logger(logs_dir: Path, *, dry_run: bool) -> tuple[logging.Logger, Path]:
    """Parallel of ``logging_setup.configure_run_logger`` with a
    ``consolidate-`` filename — same handlers, same UTC format, its own
    log family so consolidation runs are greppable apart from ingests."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-dryrun" if dry_run else ""
    log_path = logs_dir / f"consolidate-{ts}{suffix}.log"

    logger = logging.getLogger(f"brain.consolidate.{ts}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):   # avoid double-attaching if reused
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)sZ %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = _utc_converter

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(fmt)
    logger.addHandler(console_h)

    return logger, log_path


def _reindex(paths, stats: ConsolidateStats, *, logger: logging.Logger) -> None:
    """Refresh enrichment over everything the pass changed. Non-fatal at
    every step (pipeline.py's pattern): a failed refresh just means stale
    search until the next rebuild, never a failed consolidation."""
    note_paths = list(stats.touched_entity_paths)
    for old, new in stats.moved:
        # Sources are gone — upsert drops their index rows. That is the
        # point: the inbox copy must stop matching searches.
        note_paths.append(old)
        note_paths.append(new)
    if stats.digest_path:
        # F9: index the digest note we just wrote/extended, or a fresh
        # monthly digest stays unsearchable until the next full rebuild.
        note_paths.append(stats.digest_path)
    try:
        n = upsert_notes(paths, note_paths, logger=logger)
        logger.info("semantic: upserted %d chunk(s)", n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic: upsert failed (%r) — skipping", exc)
    related = None
    try:
        conn = rebuild_connections(paths, logger=logger)
        related = conn.related
    except Exception as exc:  # noqa: BLE001
        logger.warning("connections: rebuild failed (%r) — skipping", exc)
    try:
        cs = rebuild_concepts(paths, logger=logger, related=related)
        logger.info(
            "concepts: written=%d skipped=%d removed=%d",
            cs.written, cs.skipped, cs.removed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("concepts: rebuild failed (%r) — skipping", exc)
    # Promotes merge relations into entity notes — the exact data the entity
    # dashboards render — so refresh them too (every other relation-touching
    # write path does). Only worth it when a promote actually touched an entity.
    if stats.touched_entity_paths:
        try:
            db = rebuild_dashboards(paths, logger=logger)
            logger.info("dashboards: written=%d unchanged=%d", db.written, db.unchanged)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dashboards: rebuild failed (%r) — skipping", exc)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    as_of = args.as_of or datetime.now(tz=timezone.utc).date()

    paths = default_paths()
    paths.ensure()

    logger, log_path = _configure_logger(paths.logs, dry_run=args.dry_run)
    logger.info(
        "brain-consolidate start (dry_run=%s as_of=%s stale_days=%d min_confirmations=%d)",
        args.dry_run, as_of.isoformat(), args.stale_days, args.min_confirmations,
    )
    logger.info("repo root: %s", paths.root)
    logger.info("log file: %s", log_path.relative_to(paths.root))

    stats = consolidate(
        paths,
        logger=logger,
        as_of=as_of,
        stale_days=args.stale_days,
        min_confirmations=args.min_confirmations,
        dry_run=args.dry_run,
    )

    if (
        not args.dry_run
        and not args.no_reindex
        and (stats.touched_entity_paths or stats.moved)
    ):
        _reindex(paths, stats, logger=logger)

    print()
    print(
        f"Consolidation summary ({'dry-run' if args.dry_run else 'real'}, "
        f"as of {as_of.isoformat()}):"
    )
    print(f"  promoted   : {stats.promoted}")
    print(f"  digested   : {stats.digested}")
    print(f"  unresolved : {stats.unresolved}")
    print(f"  skipped    : {stats.skipped}")
    if stats.problems:
        print("  problems:")
        for p in stats.problems:
            print(f"    - {p}")
    print(f"  log        : {log_path.relative_to(paths.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
