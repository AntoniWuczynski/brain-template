#!/usr/bin/env python3
"""brain vault linter CLI.

Examples:
    uv run python scripts/sweep.py
    uv run python scripts/sweep.py --as-of 2026-06-12 --stale-days 14
    uv run python scripts/sweep.py --write-report

Always exits 0: a linter that fails the shell breaks cron pipelines —
the per-category counts (and the optional report note) are the signal.
The checks themselves live in ``scripts/ingest_lib/sweep.py``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, UTC
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly
# (i.e. without ``uv run`` having installed the package yet).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_lib.config import default_paths  # noqa: E402
from ingest_lib.notes import _atomic_write  # noqa: E402 — shared atomic write
from ingest_lib.sweep import render_report, run_sweep  # noqa: E402

_REPORT_REL = "knowledge/index/sweep-report.md"


def _configure_sweep_logger(logs_dir: Path) -> tuple[logging.Logger, Path]:
    """Per-run append-only log file (AGENTS.md rule 5). Mirrors
    ``ingest_lib.logging_setup.configure_run_logger``, which hard-codes
    the ``ingest-`` filename prefix — hence this minimal ``sweep-`` twin."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"sweep-{ts}.log"

    logger = logging.getLogger(f"brain.sweep.{ts}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):  # avoid double-attaching if reused
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)sZ %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = lambda *_args: time.gmtime()  # log timestamps in UTC

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(fmt)
    logger.addHandler(console_h)

    return logger, log_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-sweep",
        description=(
            "Lint the vault for consistency drift: orphaned archive files, "
            "dangling wikilinks, malformed/overlapping relations, fragmented "
            "concepts, stale search-index rows, and old unconsolidated "
            "assistant memory. Read-only unless --write-report is given."
        ),
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Anchor date for the staleness check and the report's updated: "
            "timestamp (default: today, UTC). Pinning it makes a sweep "
            "fully reproducible."
        ),
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=30,
        metavar="N",
        help=(
            "knowledge/assistant/ notes left memory_status: unconsolidated "
            "for more than N days (vs --as-of) are flagged (default: 30)."
        ),
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"Also write the findings to {_REPORT_REL} (atomic write).",
    )
    parser.add_argument(
        "--check-integrity",
        action="store_true",
        help="Also re-hash every archive/raw file against its recorded "
             "source_hash (archive-corrupt). Reads the whole archive (GBs), "
             "so it is off by default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    as_of: date = args.as_of or datetime.now(tz=UTC).date()

    paths = default_paths()
    paths.ensure()

    logger, log_path = _configure_sweep_logger(paths.logs)
    logger.info(
        "brain-sweep start (as_of=%s, stale_days=%d, write_report=%s)",
        as_of.isoformat(), args.stale_days, args.write_report,
    )
    logger.info("repo root: %s", paths.root)
    logger.info("log file: %s", log_path.relative_to(paths.root))

    report = run_sweep(
        paths, logger=logger, as_of=as_of, stale_days=args.stale_days,
        check_integrity=args.check_integrity,
    )

    if args.write_report:
        _atomic_write(paths.root / _REPORT_REL, render_report(report, as_of=as_of))
        logger.info("sweep: wrote report -> %s", _REPORT_REL)

    counts = report.counts
    print()
    print(f"Sweep summary (as of {as_of.isoformat()}):")
    if not counts:
        print("  no findings")
    else:
        width = max(len(c) for c in counts)
        for category, n in counts.items():
            print(f"  {category:<{width}} : {n}")
        print()
        for f in report.findings:
            print(f"  [{f.category}] {f.path} — {f.detail}")
    if args.write_report:
        print(f"  report : {_REPORT_REL}")
    print(f"  log    : {log_path.relative_to(paths.root)}")
    # Exit 0 even with findings: a linter that fails the shell breaks
    # cron pipelines; the counts above are the signal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
