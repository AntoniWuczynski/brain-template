#!/usr/bin/env python3
"""Commit tripwire: refuse a commit that adds a credential shape.

Reads the staged diff, looks only at added lines, and reports the file and
kind — never the matched value (see ``ingest_lib.sweep_checks_secrets``,
which this reuses the detection logic from). Fixtures under ``tests/``
legitimately contain fake credentials and are skipped.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly
# (mirrors scripts/sweep.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_lib.secrets import _denylist  # noqa: E402
from ingest_lib.sweep_checks_secrets import secret_kinds_in_line  # noqa: E402


def hits_in_diff(diff_text: str, denylist: list[str]) -> list[tuple[str, list[str]]]:
    """Scan a ``git diff --cached -U0`` text for added lines that contain a
    credential shape. Returns ``(path, kinds)`` pairs, one per hit line."""
    hits: list[tuple[str, list[str]]] = []
    current_file: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[len("+++ "):]
            current_file = path[2:] if path.startswith(("a/", "b/")) else path
            continue
        # "+++ " headers were consumed above, so every remaining "+" line is
        # content — including one whose own text starts with "+", such as a
        # Markdown "+ item" bullet, which a diff renders as "++ item".
        if not line.startswith("+"):
            continue
        if current_file is None or current_file.startswith("tests/"):
            continue
        kinds = secret_kinds_in_line(line[1:], denylist)
        if kinds:
            hits.append((current_file, kinds))
    return hits


def main() -> int:
    proc = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    )
    hits = hits_in_diff(proc.stdout, _denylist())
    if not hits:
        return 0

    for path, kinds in hits:
        print(f"{path}: {', '.join(kinds)}", file=sys.stderr)
    print(
        "commit blocked: possible credential(s) in staged changes above "
        "(rotate and redact, then re-stage)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
