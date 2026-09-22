#!/usr/bin/env python3
"""Report-only CLI: scan the vault for credential shapes already on disk.

Never modifies a scanned file, and never prints or writes a matched value —
only the file, line number and kind (see ``ingest_lib.sweep_checks_secrets``,
which this reuses the detection logic from).

Examples:
    uv run python scripts/secret_scan.py
    uv run python scripts/secret_scan.py --include-raw --out scan.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly
# (mirrors scripts/sweep.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_lib.secrets import _denylist  # noqa: E402
from ingest_lib.sweep_checks_secrets import secret_kinds_in_line  # noqa: E402

_TEXT_SUFFIXES = {".md", ".jsonl", ".txt"}
_SCANNED_DIRS = ("knowledge", "archive/processed", "metadata")
_MAX_RAW_BYTES = 5 * 1024 * 1024
_SNIFF_BYTES = 8192


def _is_text_decodable(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            chunk = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _iter_targets(root: Path, *, include_raw: bool) -> list[Path]:
    targets: list[Path] = []
    for area in _SCANNED_DIRS:
        base = root / area
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
                targets.append(path)

    if include_raw:
        raw_base = root / "archive/raw"
        if raw_base.exists():
            for path in sorted(raw_base.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    if path.stat().st_size > _MAX_RAW_BYTES:
                        continue
                except OSError:
                    continue
                if not _is_text_decodable(path):
                    continue
                targets.append(path)

    return targets


def scan(root: Path, *, include_raw: bool) -> list[str]:
    """Return sorted ``path:lineno: kinds`` lines for every hit."""
    denylist = _denylist()
    lines: list[str] = []
    for path in _iter_targets(root, include_raw=include_raw):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            kinds = secret_kinds_in_line(line, denylist)
            if kinds:
                lines.append(f"{rel}:{lineno}: {', '.join(kinds)}")
    return sorted(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    hits = scan(args.root, include_raw=args.include_raw)
    output_lines = [*hits, f"{len(hits)} hit(s)"]
    output = "\n".join(output_lines) + "\n"

    if args.out is not None:
        args.out.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)

    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
