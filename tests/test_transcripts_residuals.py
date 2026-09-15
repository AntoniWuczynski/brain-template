"""AUD-118 residuals: targeted regression tests for the five items the
earlier fix agent left for a follow-up pass (see
``review/2026-09-05/refute-transcripts.md`` m1/m6/n1/n2/n3), each with its
decision recorded at the point it's exercised. Kept as a sibling file to
``tests/test_transcripts.py`` per that file's own 1000-line limit.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.connectors import CONNECTORS, run_connector
from ingest_lib.connectors import _transcript_common as tc
from ingest_lib.connectors import claude_code as cc

_LOG = logging.getLogger("test")
_AT = "2026-09-03T00:00:00Z"
_A_WEEK_AGO = time.time() - 7 * 86400


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


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


def _user_line(text: str, timestamp: str, **extra: object) -> str:
    return _jsonl_line(
        type="user", timestamp=timestamp, cwd="/Users/tester/brain",
        message={"role": "user", "content": text}, **extra,
    )


def _assistant_line(text: str, timestamp: str, **extra: object) -> str:
    return _jsonl_line(
        type="assistant", timestamp=timestamp, cwd="/Users/tester/brain",
        message={"role": "assistant", "content": [{"type": "text", "text": text}]},
        **extra,
    )


# --------------------------------------------------------------------- n2
# Decision: date a session by its LAST timestamp, not its first — a
# resumed multi-day session should file/sort under when it finished, not
# under its stale start date (see the reasoning in claude_code.py's
# _parse_session).

def test_claude_code_connector_dates_session_by_last_timestamp_not_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [
        _user_line("Started this session a few days ago", "2026-08-31T09:00:00Z"),
        _assistant_line("Sure, let's start.", "2026-08-31T09:01:00Z"),
        _user_line("Picking this back up today", "2026-09-03T10:00:00Z"),
        _assistant_line("Welcome back.", "2026-09-03T10:01:00Z"),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-resumed", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snap = next((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snap.read_text())

    assert data["date"] == "2026-09-03"
    assert "2026-09-03" in snap.name
    assert "2026-08-31" not in snap.name


# --------------------------------------------------------------------- n3
# Decision: reproduced (a chmod-000 projects directory previously killed
# the whole pull with an uncaught PermissionError out of _iter_candidate_
# files) — fixed to skip rather than crash, matching the existing
# try/except OSError around f.stat() a few lines below it.

@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits only")
def test_iter_candidate_files_survives_unreadable_base_dir(tmp_path: Path) -> None:
    base = tmp_path / "unreadable_projects"
    base.mkdir()
    os.chmod(base, 0o000)
    try:
        assert cc._iter_candidate_files(base) == []
    finally:
        os.chmod(base, 0o755)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits only")
def test_claude_code_connector_pull_survives_unreadable_projects_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    projects.mkdir()
    os.chmod(projects, 0o000)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))
    try:
        paths = _vault(tmp_path)
        stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
        assert stats.written == 0
    finally:
        os.chmod(projects, 0o755)


# --------------------------------------------------------------------- m6
# Decision: leave the over-redaction as-is. Pinning both the accepted
# false positives AND the two rows that rule out the obvious fix, so a
# future change that "fixes" one of these without checking the other two
# tables fails loudly here instead of silently reopening C2/the Slack
# pinned case.

def test_redact_env_secret_false_positive_on_negated_name_is_accepted() -> None:
    # `NOT_A_SECRET` contains the substring `SECRET` — over-redaction,
    # not a leak, and the accepted trade-off (module docstring).
    cleaned, n = tc.redact_secrets("NOT_A_SECRET=plainvalue")
    assert n == 1
    assert "[REDACTED:env-secret]" in cleaned


def test_redact_slack_token_false_positive_on_documentation_string_is_accepted() -> None:
    cleaned, n = tc.redact_secrets("xoxb-not-a-real-token-this-is-documentation")
    assert n == 1
    assert "[REDACTED:slack-token]" in cleaned


@pytest.mark.parametrize(
    "raw",
    [
        # No digit anywhere in the value — a digit-required fix (the
        # rejected discriminator) would un-redact these, reopening C2.
        "TOKEN=supersecretvalue",
        "SECRET=supersecretvalue",
    ],
)
def test_redact_env_secret_bare_keyword_has_no_digit_and_still_must_redact(raw: str) -> None:
    cleaned, n = tc.redact_secrets(raw)
    assert n == 1
    assert "[REDACTED:env-secret]" in cleaned


def test_redact_slack_token_pinned_case_has_no_digit_and_still_must_redact() -> None:
    # Matches the pinned row in test_redact_secrets_known_patterns exactly
    # — a digit-required fix would un-redact this too.
    raw = "xoxb-" + "a" * 15
    cleaned, n = tc.redact_secrets(f"here is a secret: {raw} end")
    assert n == 1
    assert "[REDACTED:slack-token]" in cleaned
