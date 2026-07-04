"""Append-only JSONL telemetry for tool calls.

Two streams under ``logs/`` (gitignored — telemetry is per-machine and
regenerable in spirit, never vault content):

- ``logs/mcp-audit.jsonl``  — writes and tool outcomes (who changed what).
- ``logs/mcp-access.jsonl`` — reads and searches (who looked at what).

Design constraints:

- **Fail-open.** Logging must never break a tool call: every OSError is
  swallowed after a log.warning. A full disk should degrade telemetry,
  not writes to the vault.
- **Open-append-close per event.** Event rates are tiny (the write rate
  limit caps tools at well under ~150 events/min), so durability beats
  keeping a file handle hot: nothing to flush on crash, log rotation
  needs no coordination.
- **One JSON object per line**, sorted keys, ensure_ascii=False so the
  files stay greppable and jq-able. ``json.dumps`` escapes embedded
  newlines, so a row can never span lines.

Timestamps live here deliberately: AGENTS.md confines timestamps to
frontmatter and logs, and this *is* a log.
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _now_iso() -> str:
    """ISO-8601 UTC with a Z suffix, matching vault frontmatter style."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditLog:
    """Thread-safe appender for the two telemetry streams."""

    def __init__(self, vault_root: Path) -> None:
        self._audit_path = vault_root / "logs" / "mcp-audit.jsonl"
        self._access_path = vault_root / "logs" / "mcp-access.jsonl"
        # Tool calls run concurrently in a threadpool; the lock keeps two
        # events from interleaving mid-line. One lock for both files is
        # fine at these rates.
        self._lock = threading.Lock()

    def tool_event(
        self,
        *,
        agent: str,
        tool: str,
        path: str | None,
        outcome: str,
        detail: str | None = None,
    ) -> None:
        """Record a write / mutating tool call and how it ended."""
        self._append(self._audit_path, {
            "ts": _now_iso(),
            "agent": agent,
            "tool": tool,
            "path": path,
            "outcome": outcome,
            "detail": detail,
        })

    def access_event(
        self,
        *,
        agent: str,
        tool: str,
        paths: Sequence[str],
        query: str | None = None,
    ) -> None:
        """Record a read / search and which vault paths it touched."""
        self._append(self._access_path, {
            "ts": _now_iso(),
            "agent": agent,
            "tool": tool,
            "paths": list(paths),
            "query": query,
        })

    def _append(self, path: Path, row: dict[str, object]) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        try:
            with self._lock:
                # logs/ exists in a real vault; tolerate fresh test roots.
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            # Fail-open: telemetry loss is acceptable, a broken tool call
            # is not. The warning lands in the server log only.
            log.warning("audit: could not append to %s: %s", path.name, exc)
