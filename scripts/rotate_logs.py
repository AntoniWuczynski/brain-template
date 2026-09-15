#!/usr/bin/env python3
"""brain log-rotation CLI.

Examples:
    uv run python scripts/rotate_logs.py --dry-run
    uv run python scripts/rotate_logs.py
    uv run python scripts/rotate_logs.py --max-mb 5

Two jobs, one retention policy: **compress and keep, never delete.**

1. Rotates the two MCP telemetry streams written by ``mcp_server/audit.py``
   (``logs/mcp-access.jsonl`` and ``logs/mcp-audit.jsonl``) once they cross
   a size threshold. Nothing else rotates them — the writer only ever
   appends, so left alone these files grow forever.
2. Gzips the dated per-run logs (``ingest-<stamp>.log``, ``sweep-…``,
   ``dream-…``, ``consolidate-…`` and anything else named
   ``<tool>-YYYYmmddTHHMMSSZ[-suffix].log``) once they are older than
   ``--compress-after-days``. Each becomes ``<name>.log.gz`` in place, kept
   forever. The scheduled jobs write one such file per run and nothing ever
   removed them (229+ of them, a few hundred bytes each, on this machine).

Retention policy, both jobs: nothing is ever pruned or truncated. A log is
either active, or an immutable ``.gz`` that stays until a human deletes it.
Compression is the only lifecycle step, so "how far back do the logs go" has
the same answer forever: all the way.

Deliberately NOT rotated: the launchd/nohup stdout streams
(``brain-mcp.launchd.log``, ``maintain.out.log``/``.err.log``,
``dream.out.log``/``.err.log``). launchd holds those open for the life of
the process, so renaming one leaves the running server appending to a
now-unlinked inode — every later line silently lost until a restart. They
need copy-truncate (or a restart) rather than the rename-and-gzip below;
until that is built, they are the operator's business, not this script's.

Benign race, no locking needed: ``AuditLog._append`` opens, appends, and
closes the target file per event (see ``mcp_server/audit.py``'s docstring:
"log rotation needs no coordination"), and this CLI is meant to run from
``scripts/maintain.sh`` in a quiet hour. A concurrent append during a
rotation lands either in the fresh active file the writer's next
``open(..., "a")`` call recreates, or in the just-renamed segment. One
microscopic window exists — an append whose ``open`` beat the rename but
whose write lands after the segment is unlinked goes to a dead inode —
and that is acceptable here: audit.py is fail-open by contract, so a
telemetry event may be lost, vault writes never.

Crash safety: the gzip is written to a ``.tmp`` and atomically renamed
into place, and every run first finishes any rotation a previous run
left half done (a renamed ``mcp-*-<ts>.jsonl`` segment that never got
compressed), so an interruption never strands raw telemetry.

Rotated ``.gz`` segments are kept forever (retention: keep-forever) and
this script never touches them again once written; ``.gitignore`` already
excludes ``logs/*.gz`` (and the two active ``.jsonl`` targets) from git.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly
# (i.e. without ``uv run`` having installed the package yet).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_lib.config import default_paths  # noqa: E402

_DEFAULT_MAX_MB = 10
_TARGET_NAMES = ("mcp-access.jsonl", "mcp-audit.jsonl")

_DEFAULT_COMPRESS_DAYS = 14
# A per-run log names itself ``<tool>-<UTC stamp>[-suffix].log`` (AGENTS.md
# hard rule 5: ``logs/ingest-YYYYMMDDTHHMMSS.log`` "or a similarly named log
# for other tools"). Matching the stamp rather than a list of tool names
# covers future tools for free, and — the reason it is a stamp match and not
# ``*.log`` — excludes the stdout streams launchd holds open
# (``brain-mcp.launchd.log``, ``maintain.out.log``, …), which carry no stamp
# and must not be renamed underneath a live writer.
_RUN_LOG_RE = re.compile(r"^.+-(?P<stamp>\d{8}T\d{6}Z)(?:-[^/]*)?\.log$")


def _now_stamp() -> str:
    """UTC timestamp for segment names; separate so tests can pin it."""
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _gzip_segment(segment: Path, gz_path: Path) -> None:
    """Stream-gzip ``segment`` to ``gz_path`` (atomic: .tmp then rename),
    then unlink the uncompressed segment. Crash-safe at every step: a
    partial ``.tmp`` is overwritten on retry, and a completed ``gz_path``
    is only ever renamed into place whole.

    The unlink drops the last uncompressed copy, so the compressed bytes
    and the rename that publishes them are fsynced first (same discipline
    as ``notes._atomic_write``) — otherwise a crash can leave a truncated
    or absent ``.gz`` as the only remaining copy of a keep-forever
    segment."""
    tmp = gz_path.with_name(gz_path.name + ".tmp")
    with segment.open("rb") as src, gzip.open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    tmp.replace(gz_path)
    dir_fd = os.open(gz_path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    segment.unlink()


def finish_orphaned_segments(logs_dir: Path, *, dry_run: bool = False) -> list[Path]:
    """Finish rotations a previous run left half done.

    A crash between rename and gzip leaves an uncompressed
    ``mcp-<stream>-<ts>.jsonl`` segment behind (raw telemetry, not
    gitignored the way ``.gz`` segments are). Compress and remove any
    such segment; if its ``.gz`` already exists the compression finished
    and only the unlink was lost, so just unlink. Returns the ``.gz``
    paths (that would be) produced or completed."""
    finished: list[Path] = []
    for name in _TARGET_NAMES:
        stem = Path(name).stem
        for segment in sorted(logs_dir.glob(f"{stem}-*.jsonl")):
            gz_path = segment.with_name(segment.name + ".gz")
            finished.append(gz_path)
            if dry_run:
                continue
            if gz_path.exists():
                segment.unlink()
            else:
                _gzip_segment(segment, gz_path)
    return finished


def compress_run_logs(
    logs_dir: Path,
    *,
    older_than_days: int,
    now: datetime | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Gzip dated per-run logs older than ``older_than_days``, in place.

    A run log is any ``<tool>-<UTC stamp>[-suffix].log`` in ``logs_dir``; its
    age comes from the stamp in its own name, not from mtime, so the result
    is deterministic and a touched file doesn't get a stay of execution. Each
    qualifying log becomes ``<name>.log.gz`` (same atomic tmp-then-rename,
    fsync-before-unlink discipline as the telemetry segments) and is then
    kept forever — nothing here deletes a log, compressed or not.

    Recovery, same as ``finish_orphaned_segments``: if the ``.gz`` already
    exists, a previous run completed the compression and lost only the
    unlink, so drop the redundant plain copy.

    Returns the ``.gz`` paths (that would be) produced, sorted by name."""
    if now is None:
        now = datetime.now(tz=UTC)
    cutoff = now - timedelta(days=older_than_days)
    produced: list[Path] = []
    for log_file in sorted(logs_dir.glob("*.log")):
        match = _RUN_LOG_RE.match(log_file.name)
        if match is None:
            continue
        try:
            stamped = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue  # a 99991301T… lookalike: not a stamp, leave it alone
        if stamped > cutoff:
            continue
        gz_path = log_file.with_name(log_file.name + ".gz")
        produced.append(gz_path)
        if dry_run:
            continue
        if gz_path.exists():
            log_file.unlink()
        else:
            _gzip_segment(log_file, gz_path)
    return produced


def rotate_if_large(path: Path, max_bytes: int, *, dry_run: bool = False) -> Path | None:
    """Rotate ``path`` to a timestamped, gzipped segment if it exceeds ``max_bytes``.

    No-op (returns ``None``) if ``path`` doesn't exist or its size is at or
    under ``max_bytes``. Otherwise: rename ``path`` to
    ``<stem>-<UTC ts YYYYMMDDTHHMMSSZ><suffix>`` in the same directory,
    stream-gzip that renamed file to the same name plus ``.gz``, unlink the
    uncompressed renamed copy, and return the ``.gz`` path. ``path`` itself
    is left absent afterwards — ``mcp_server/audit.py`` recreates it on its
    next append.

    ``dry_run=True`` performs no filesystem writes and returns the ``.gz``
    path that a real run would produce (it is not created on disk).
    """
    if not path.exists():
        return None
    if path.stat().st_size <= max_bytes:
        return None

    ts = _now_stamp()
    rotated = path.with_name(f"{path.stem}-{ts}{path.suffix}")
    gz_path = rotated.with_name(f"{rotated.name}.gz")

    if dry_run:
        return gz_path

    if rotated.exists() or gz_path.exists():
        # Same-second name collision — only reachable via duplicate or
        # concurrent scheduling. Never overwrite an existing segment;
        # skip, and a later run rotates under a fresh timestamp.
        logging.getLogger("brain.rotate_logs").warning(
            "segment %s already exists; skipping rotation of %s", rotated.name, path.name
        )
        return None

    path.rename(rotated)
    _gzip_segment(rotated, gz_path)
    return gz_path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-rotate-logs",
        description=(
            "Rotate logs/mcp-access.jsonl and logs/mcp-audit.jsonl once they "
            "cross a size threshold, and gzip dated per-run logs once they "
            "are old enough. Compressed segments are kept forever, never "
            "pruned. Deterministic, exit 0 on success — cron/launchd-safe."
        ),
    )
    parser.add_argument(
        "--max-mb",
        type=int,
        default=_env_int("BRAIN_LOG_ROTATE_MB", _DEFAULT_MAX_MB),
        metavar="N",
        help="Rotate a target once it exceeds N MiB (default: 10, env BRAIN_LOG_ROTATE_MB).",
    )
    parser.add_argument(
        "--compress-after-days",
        type=int,
        default=_env_int("BRAIN_LOG_COMPRESS_DAYS", _DEFAULT_COMPRESS_DAYS),
        metavar="N",
        help=(
            "Gzip a dated per-run log once it is N days old, keeping it "
            "forever (default: 14, env BRAIN_LOG_COMPRESS_DAYS)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would rotate; write nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    logger = logging.getLogger("brain.rotate_logs")
    max_bytes = args.max_mb * 1024 * 1024

    paths = default_paths()
    try:
        paths.ensure()
        for gz in finish_orphaned_segments(paths.logs, dry_run=args.dry_run):
            verb = "would finish" if args.dry_run else "finished"
            print(f"rotate-logs: {verb} orphaned segment -> {gz.relative_to(paths.root)}")
        compressed = compress_run_logs(
            paths.logs, older_than_days=args.compress_after_days, dry_run=args.dry_run
        )
        if compressed:
            verb = "would compress" if args.dry_run else "compressed"
            print(
                f"rotate-logs: {verb} {len(compressed)} run log(s) older than "
                f"{args.compress_after_days} day(s), kept as .gz"
            )
        else:
            logger.info("run logs: none older than %s day(s)", args.compress_after_days)
        rotated_any = False
        for name in _TARGET_NAMES:
            target = paths.logs / name
            result = rotate_if_large(target, max_bytes, dry_run=args.dry_run)
            if result is None:
                logger.info("%s: below threshold or absent", name)
                continue
            rotated_any = True
            verb = "would rotate" if args.dry_run else "rotated"
            print(f"rotate-logs: {name}: {verb} -> {result.relative_to(paths.root)}")
        if not rotated_any and not compressed:
            print("rotate-logs: nothing to rotate")
        return 0
    except OSError as exc:
        logger.error("rotate-logs failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
