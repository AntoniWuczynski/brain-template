#!/usr/bin/env python3
"""Pull an external source into the vault as archivable snapshots.

Examples:
    uv run python scripts/pull.py --list
    uv run python scripts/pull.py <connector> --dry-run
    uv run python scripts/pull.py <connector> --then-ingest

A connector fetches new/changed items from its source and writes each as a
snapshot under ``inbox/<source_class>/``; the normal ingest pipeline then
copies them to the immutable archive and extracts them. Idempotent: an
unchanged item is skipped before it touches ``inbox/`` (per-connector state
in ``metadata/connectors/``).

Exits 0 on a clean run so it is cron/launchd-safe, like sweep/consolidate.
Secrets (API tokens) come from the environment / ``.env`` — never a flag.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = Path(__file__).resolve().parent.parent

from ingest_lib.config import default_paths  # noqa: E402
from ingest_lib.connectors import CONNECTORS, run_connector  # noqa: E402


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)


def _configure_logger(logs_dir: Path, name: str) -> logging.Logger:
    """Per-run append-only log (AGENTS.md rule 5), mirroring sweep.py's."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    logger = logging.getLogger(f"brain.pull.{name}.{ts}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fmt.converter = lambda *_a: time.gmtime()
    file_h = logging.FileHandler(logs_dir / f"pull-{name}-{ts}.log", encoding="utf-8")
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(fmt)
    logger.addHandler(console_h)
    return logger


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brain-pull", description=__doc__)
    ap.add_argument("connector", nargs="?", help="connector name (see --list)")
    ap.add_argument("--list", action="store_true", help="list registered connectors")
    ap.add_argument("--dry-run", action="store_true", help="report what would be pulled, write nothing")
    ap.add_argument("--then-ingest", action="store_true",
                    help="run the ingest pipeline over inbox/ after pulling")
    args = ap.parse_args(argv)

    if args.list or not args.connector:
        names = sorted(CONNECTORS)
        if names:
            print("registered connectors:")
            for n in names:
                print(f"  {n}")
        else:
            print("no connectors registered yet — see scripts/ingest_lib/connectors/.")
        return 0

    if args.connector not in CONNECTORS:
        print(f"unknown connector: {args.connector!r} (see --list)", file=sys.stderr)
        return 2

    _load_env()
    paths = default_paths()
    paths.ensure()
    logger = _configure_logger(paths.logs, args.connector)
    connector = CONNECTORS[args.connector]()
    pulled_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    stats = run_connector(connector, paths, pulled_at=pulled_at,
                          dry_run=args.dry_run, logger=logger)
    print(f"pull {args.connector}: written={stats.written} skipped={stats.skipped}"
          f"{' (dry-run)' if args.dry_run else ''}")

    if args.then_ingest and not args.dry_run and stats.written:
        from ingest_lib.pipeline import plan_ingest, run_ingest
        plan = plan_ingest(paths, sources=[paths.inbox], from_archive=False, logger=logger)
        ing = run_ingest(paths, plan, dry_run=False, logger=logger)
        print(f"ingest: processed={ing.processed} partial={ing.partial} "
              f"manual_review={ing.manual_review} skipped={ing.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
