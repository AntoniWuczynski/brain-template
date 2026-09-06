"""Tests for mcp_server.audit — append-only JSONL telemetry.

Constraints under test: rows are parseable single-line JSON with sorted
keys, concurrent writers never interleave mid-line, and an OSError from
the filesystem is swallowed (fail-open: telemetry must never break a
tool call).
"""
from __future__ import annotations

import json
import logging
import os
import stat
import sys
import threading
from pathlib import Path
from typing import NoReturn

import pytest

# mcp_server is not an installed package (only ingest_lib is). The full
# suite imports it via a collection-order side effect; pin the repo root
# onto sys.path so this file also runs standalone.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp_server.audit import AuditLog  # noqa: E402


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_tool_event_lands_as_sorted_jsonl(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path)
    audit.tool_event(
        agent="agent-a", tool="vault_create_note",
        path="knowledge/notes/x.md", outcome="ok", detail="created",
    )
    audit.tool_event(
        agent="agent-b", tool="vault_replace_note",
        path=None, outcome="denied",
    )

    lines = _lines(tmp_path / "logs" / "mcp-audit.jsonl")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["agent"] == "agent-a"
    assert first["tool"] == "vault_create_note"
    assert first["path"] == "knowledge/notes/x.md"
    assert first["outcome"] == "ok"
    assert first["detail"] == "created"
    assert first["ts"].endswith("Z")
    # sort_keys=True: key order on disk is deterministic.
    assert list(first.keys()) == sorted(first.keys())
    second = json.loads(lines[1])
    assert second["path"] is None and second["detail"] is None


def test_access_event_lands_in_access_log(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path)
    audit.access_event(
        agent="agent-a", tool="vault_search",
        paths=["archive/processed/a.md", "knowledge/notes/b.md"],
        query="zażółć gęślą jaźń",  # ensure_ascii=False survives round-trip
    )
    audit.access_event(agent="agent-a", tool="vault_read", paths=["knowledge/notes/b.md"])

    audit_path = tmp_path / "logs" / "mcp-audit.jsonl"
    access_path = tmp_path / "logs" / "mcp-access.jsonl"
    assert not audit_path.exists()  # reads never pollute the write log
    lines = _lines(access_path)
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["paths"] == ["archive/processed/a.md", "knowledge/notes/b.md"]
    assert first["query"] == "zażółć gęślą jaźń"
    assert "zażółć" in lines[0]  # not \u-escaped
    assert json.loads(lines[1])["query"] is None


def test_concurrent_writers_do_not_interleave(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path)
    n_threads, per_thread = 8, 50

    def writer(idx: int) -> None:
        for i in range(per_thread):
            audit.tool_event(
                agent=f"agent-{idx}", tool="vault_append_to_note",
                path=f"knowledge/notes/{idx}-{i}.md", outcome="ok",
                detail="x" * 200,  # long enough that a torn write would show
            )

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = _lines(tmp_path / "logs" / "mcp-audit.jsonl")
    assert len(lines) == n_threads * per_thread
    # Every line must parse on its own — a mid-line interleave would break this.
    for line in lines:
        row = json.loads(line)
        assert row["outcome"] == "ok"


def test_oserror_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    audit = AuditLog(tmp_path)
    attempts: list[str] = []

    def boom(path: object, *args: object, **kwargs: object) -> NoReturn:
        attempts.append(Path(str(path)).name)
        raise OSError("disk full")

    monkeypatch.setattr(os, "open", boom)
    with caplog.at_level(logging.WARNING, logger="mcp_server.audit"):
        # Must not raise: logging failure never propagates into the tool call.
        audit.tool_event(agent="a", tool="t", path=None, outcome="ok")
        audit.access_event(agent="a", tool="t", paths=[])

    # AUD-072: "did not raise" is not evidence the fail-open branch ran — a
    # writer that stopped calling the patched name would pass that too. Pin
    # the branch itself: both appends reached the opener, both failed, and
    # both were reported as a warning rather than swallowed in silence.
    assert attempts == ["mcp-audit.jsonl", "mcp-access.jsonl"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("disk full" in w for w in warnings)
    assert [w for w in warnings if "mcp-audit.jsonl" in w]
    assert [w for w in warnings if "mcp-access.jsonl" in w]
    # ...and nothing landed on disk, so the rows really were lost.
    assert not (tmp_path / "logs" / "mcp-audit.jsonl").exists()
    assert not (tmp_path / "logs" / "mcp-access.jsonl").exists()


def test_new_log_file_created_with_0600(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path)
    audit.tool_event(agent="a", tool="t", path=None, outcome="ok")

    path = tmp_path / "logs" / "mcp-audit.jsonl"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
