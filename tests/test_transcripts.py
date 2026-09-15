"""Chat-transcript connectors (Claude Code sessions, claude.ai/ChatGPT
conversation exports) + the shared transcript extractor.

This module's tests outgrew the owner's 1000-line-per-file rule and were
split along their existing section boundaries into sibling modules:

- ``test_transcripts_extractor.py`` — the shared transcript extractor and
  its connector/extractor registration.
- ``test_transcripts_redaction.py`` — ``_transcript_common`` secret
  redaction and turn finalisation.
- ``test_transcripts_claude_code.py`` — the ``claude_code`` connector.
- ``test_transcripts_chat_export.py`` — the ``chat_export`` connector and
  ``scripts/pull.py``'s ``--path`` flag.

(``test_transcripts_residuals.py`` is a separate, pre-existing sibling for
a different follow-up pass — not part of this split.)

Nothing outside this file imported it, but the helper names it used to
define are kept bound here below (matching how ``test_transcripts_residuals.py``
keeps its own copies rather than reaching across test modules — a bare
cross-file import between sibling test modules resolves fine for pytest's
own collection, which prepends each file's directory to ``sys.path``, but
not for mypy, which has no ``__init__.py`` chain to hang a package name on)
for anyone (or any monkeypatch) that still reaches for them by their old
``test_transcripts.*`` path.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root

_LOG = logging.getLogger("test")
_AT = "2026-09-03T00:00:00Z"


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def _snapshot(**kw: object) -> bytes:
    return json.dumps(kw).encode("utf-8")


def _jsonl_line(**kw: object) -> str:
    return json.dumps(kw) + "\n"


def _write_session(
    projects_dir: Path, project_slug: str, session_id: str, lines: list[str],
    *, mtime: float | None = None,
) -> Path:
    proj = projects_dir / project_slug
    proj.mkdir(parents=True, exist_ok=True)
    f = proj / f"{session_id}.jsonl"
    f.write_text("".join(lines), encoding="utf-8")
    if mtime is not None:
        os.utime(f, (mtime, mtime))
    return f


def _user_line(text: str | list[object], **extra: object) -> str:
    return _jsonl_line(
        type="user", timestamp="2026-09-03T10:00:00Z", cwd="/Users/tester/brain",
        message={"role": "user", "content": text}, **extra,
    )


def _assistant_line(text: str | list[object], **extra: object) -> str:
    return _jsonl_line(
        type="assistant", timestamp="2026-09-03T10:01:00Z", cwd="/Users/tester/brain",
        message={"role": "assistant", "content": text}, **extra,
    )


_A_WEEK_AGO = time.time() - 7 * 86400

__all__ = [
    "_A_WEEK_AGO",
    "_AT",
    "_LOG",
    "_assistant_line",
    "_jsonl_line",
    "_snapshot",
    "_user_line",
    "_vault",
    "_write_session",
]
