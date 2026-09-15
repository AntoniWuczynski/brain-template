"""Shared pieces for the ``sweep_checks_*`` modules.

Pulled out of ``sweep.py`` because every check module needs the
``Finding`` type and the two tolerant-read helpers below, and a sibling
module must not import back into ``sweep.py`` itself (that would make
``sweep.py``'s re-exports of these very names into an import cycle)."""
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    """One lint finding. ``path`` is vault-relative; ``detail`` is a
    human-readable, deterministic one-liner."""

    category: str
    path: str
    detail: str


def _read_text(path: Path) -> str | None:
    """Tolerant read: a vanished or unreadable file is a skip, not a crash."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ``git check-ignore`` answers "is this path excluded by the repository's
# ignore rules?" from .gitignore itself, so no second list of expected-absent
# sources has to be maintained alongside it and the two can never drift.
_GIT_CHECK_IGNORE_TIMEOUT_S = 30


def _git_ignored_paths(root: Path, rels: Sequence[str]) -> frozenset[str]:
    """The subset of ``rels`` (root-relative POSIX paths) that git reports
    as excluded by the vault's ignore rules.

    One batched ``git check-ignore`` call, NUL-delimited in both directions
    so spaces and other odd characters in slide filenames survive. Tracked
    paths are never reported ignored (git consults the index first), so a
    committed file that vanished stays a real finding.

    Returns an empty set whenever the answer cannot be obtained — git not
    installed, the vault not a repository, the call timing out. The honest
    default is to keep reporting an absent file as missing rather than to
    excuse it on a guess.
    """
    if not rels:
        return frozenset()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--stdin", "-z"],
            input="\0".join(rels),
            capture_output=True,
            text=True,
            timeout=_GIT_CHECK_IGNORE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    # 0 = at least one path ignored, 1 = none ignored. Anything else (128 =
    # not a repository) means the output carries no answer.
    if proc.returncode not in (0, 1):
        return frozenset()
    return frozenset(line for line in proc.stdout.split("\0") if line)
