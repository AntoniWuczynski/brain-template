"""Per-run log file + console handler. Each ingestion run gets its own file."""
from __future__ import annotations

import logging
import time
from datetime import datetime, UTC
from pathlib import Path


def configure_run_logger(logs_dir: Path, *, dry_run: bool) -> tuple[logging.Logger, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-dryrun" if dry_run else ""
    log_path = logs_dir / f"ingest-{ts}{suffix}.log"

    logger = logging.getLogger(f"brain.ingest.{ts}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Avoid double-attaching if reused.
    for h in list(logger.handlers):
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


def _utc_converter(timestamp: float | None = None) -> time.struct_time:
    # Honor the record's creation timestamp (not the formatting moment), so a
    # buffered/delayed flush stamps event-time rather than flush-time.
    return time.gmtime(timestamp)
