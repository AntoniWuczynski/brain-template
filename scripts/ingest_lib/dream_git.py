"""Git subprocess helper shared by the dream-pass gate, packet and
compiled-truth modules.

Split out of ``dream.py`` (AUD-121): the lowest-level piece of the dream
machinery, used by ``dream_gate_core``, ``dream_compiled`` and
``dream_packet`` alike, so it lives underneath all three rather than in
any one of them.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """A git subprocess failed or timed out."""


def _git(root: Path, *args: str, timeout: float = 30.0) -> str:
    """Run git with explicit argv (no shell), mirroring mcp_server/git_ops."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # `from None`, and stderr capped, for the same reason as git_ops:
        # this text reaches the scheduled-run log via dream_gate, and raw git
        # stderr can carry remote URLs / ssh hints.
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from None
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()[:300]}")
    return result.stdout
