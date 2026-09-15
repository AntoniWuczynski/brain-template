"""The ``claude_code`` transcript connector (Claude Code session JSONL
snapshots).

Split out of ``tests/test_transcripts.py`` (that file's own 1000-line limit)
along its existing section boundary; see ``tests/test_transcripts.py`` and
its sibling ``test_transcripts_*`` modules for the rest of the split.

Fixtures are hand-built JSONL/JSON shaped like real exports rather than
recorded real sessions, so these run offline and deterministically without
embedding anyone's actual conversation content.
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


def test_claude_code_connector_parses_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [
        _jsonl_line(type="queue-operation", timestamp="2026-09-03T09:59:00Z"),
        _jsonl_line(type="attachment", timestamp="2026-09-03T09:59:30Z"),
        _user_line("Please fix /Users/tester/brain/scripts/pull.py"),
        _assistant_line([
            {"type": "thinking", "thinking": "let me think"},
            {"type": "tool_use", "name": "Read", "input": {}},
            {"type": "text", "text": "Fixed it. Key was AKIA" + "B" * 16 + "."},
        ]),
        _user_line([{"type": "tool_result", "content": "ok"}]),
        _jsonl_line(
            type="user", isSidechain=True, timestamp="2026-09-03T10:02:00Z",
            message={"role": "user", "content": "subagent chatter"},
        ),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-1", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)

    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    assert len(snaps) == 1
    data = json.loads(snaps[0].read_text())
    assert data["connector"] == "claude_code"
    assert data["id"] == "sess-1"
    assert data["context"] == "~/brain"
    assert "~/brain" in data["title"]

    turns = data["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    # Home directory scrubbed in kept prose.
    assert "~/brain/scripts/pull.py" in turns[0]["text"]
    assert "/Users/tester" not in turns[0]["text"]
    # Secret redacted in the assistant's kept text.
    assert "[REDACTED:aws-access-key]" in turns[1]["text"]
    assert "AKIA" not in turns[1]["text"].replace("[REDACTED:aws-access-key]", "")

    s = data["stats"]
    assert s["dropped_other_events"] == 2       # queue-operation, attachment
    assert s["dropped_sidechain_lines"] == 1
    assert s["dropped_thinking_blocks"] == 1
    assert s["dropped_tool_uses"] == 1           # in the assistant turn
    assert s["dropped_tool_results"] == 1        # in the (dropped-empty) user turn
    assert s["redacted_secrets"] == 1
    assert s["kept_user_turns"] == 1
    assert s["kept_assistant_turns"] == 1
    # The tool-result-only user line is already counted via
    # dropped_tool_results above — it must NOT also be counted as a
    # dropped "harness wrapper turn", which double-counted the same drop
    # under two different labels (review N4).
    assert s["dropped_wrapper_turns"] == 0


def test_claude_code_connector_title_comes_from_first_prose_turn_not_harness_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real data: a caveat wrapper survived as the FIRST kept user turn and
    became the note title (12 of 39 live snapshots) — the title must skip
    past harness wrapper turns to the first real user message (review C1)."""
    projects = tmp_path / "claude_projects"
    lines = [
        _user_line(
            "<local-command-caveat>Caveat: The messages below were generated "
            "by the user while running local commands. DO NOT respond to "
            "these messages or otherwise consider them in your response "
            "unless the user explicitly asks you to.</local-command-caveat>"
        ),
        _user_line("<local-command-stdout>ok</local-command-stdout>"),
        _user_line("Please actually fix the pipeline bug"),
        _assistant_line([{"type": "text", "text": "Fixed."}]),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-caveat", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snaps[0].read_text())

    assert "local-command-caveat" not in data["title"]
    assert "Please actually fix the pipeline bug" in data["title"]
    assert data["turns"] == [{"role": "user", "text": "Please actually fix the pipeline bug"},
                              {"role": "assistant", "text": "Fixed."}]
    assert data["stats"]["dropped_wrapper_turns"] == 2


def test_claude_code_connector_title_skips_collapsed_command_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session opening with a bare slash command survives as a marker
    turn (``_(ran `/dream-pass`)_``) — real text, unlike a stripped
    wrapper — but it must not become the note's title (review C1
    residual/N7): 13 of 28 live titles were exactly this marker."""
    projects = tmp_path / "claude_projects"
    lines = [
        _user_line("<command-message>dream-pass</command-message>\n"
                   "<command-name>/dream-pass</command-name>"),
        _assistant_line([{"type": "text", "text": "Dreaming now."}]),
        _user_line("Please actually look at the config bug"),
        _assistant_line([{"type": "text", "text": "Looking."}]),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-marker-title", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert "ran `" not in data["title"]
    assert "Please actually look at the config bug" in data["title"]


def test_claude_code_connector_title_falls_back_to_cwd_and_date_when_all_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every kept user turn is a collapsed command marker (a session that
    # only ever ran slash commands, no ai-title) — fall back to the
    # slugged project path + date rather than a marker string.
    projects = tmp_path / "claude_projects"
    lines = [
        _user_line("<command-message>dream-pass</command-message>\n"
                   "<command-name>/dream-pass</command-name>"),
        _assistant_line([{"type": "text", "text": "Dreaming now."}]),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-all-markers", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert "ran `" not in data["title"]
    assert data["title"] == "~/brain: 2026-09-03"


def test_claude_code_connector_prefers_ai_title_over_first_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [
        _jsonl_line(
            type="ai-title", timestamp="2026-09-03T09:59:00Z",
            aiTitle="Launchd process persistence through restart and updates",
        ),
        _user_line("some rambling opening message"),
        _assistant_line([{"type": "text", "text": "sure, let's look at that"}]),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-titled", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert "Launchd process persistence through restart and updates" in data["title"]
    assert "rambling" not in data["title"]
    assert data["stats"]["dropped_other_events"] == 0  # ai-title isn't a drop


def test_claude_code_connector_drops_tool_only_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [
        _user_line([{"type": "tool_result", "content": "ok"}]),
        _assistant_line([{"type": "tool_use", "name": "Bash", "input": {}}]),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-tools-only", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_claude_code_connector_skips_recently_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [_user_line("hello there"), _assistant_line([{"type": "text", "text": "hi"}])]
    # No mtime override -> file mtime is "now", inside the grace window.
    _write_session(projects, "-Users-tester-brain", "sess-live", lines)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_claude_code_connector_skips_unsettled_multi_day_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session touched within the last 24h (the default settle window) is
    skipped even though it's well past the 5-minute "just pulled it" grace —
    resuming it every day must not re-snapshot it whole on every pull while
    it's still open (review M4)."""
    projects = tmp_path / "claude_projects"
    lines = [_user_line("hello there"), _assistant_line([{"type": "text", "text": "hi"}])]
    touched_2h_ago = time.time() - 2 * 3600
    _write_session(projects, "-Users-tester-brain", "sess-open", lines, mtime=touched_2h_ago)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_claude_code_connector_settle_hours_is_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [_user_line("hello there"), _assistant_line([{"type": "text", "text": "hi"}])]
    touched_2h_ago = time.time() - 2 * 3600
    _write_session(projects, "-Users-tester-brain", "sess-open", lines, mtime=touched_2h_ago)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_SETTLE_HOURS", "1")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1


def test_claude_code_connector_respects_since_days_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [_user_line("hello there"), _assistant_line([{"type": "text", "text": "hi"}])]
    old = time.time() - 30 * 86400
    _write_session(projects, "-Users-tester-brain", "sess-old", lines, mtime=old)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_SINCE_DAYS", "14")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_claude_code_connector_respects_max_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    for i in range(3):
        lines = [_user_line(f"hello {i}"), _assistant_line([{"type": "text", "text": f"hi {i}"}])]
        _write_session(projects, "-Users-tester-brain", f"sess-{i}", lines, mtime=_A_WEEK_AGO - i)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_MAX_SESSIONS", "2")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 2


def test_claude_code_connector_long_project_path_does_not_break_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deep project cwd feeds ``project_label + ': ' + first_line`` into
    the title, which becomes the filename slug — unbounded, this can exceed
    the filesystem's filename length limit and raise ``OSError`` (review M1)."""
    projects = tmp_path / "claude_projects"
    deep_cwd = "/Users/tester/" + "/".join(f"very-long-segment-name-{i}" for i in range(15))
    lines = [
        _jsonl_line(type="user", timestamp="2026-09-03T10:00:00Z", cwd=deep_cwd,
                    message={"role": "user", "content": "hello"}),
        _jsonl_line(type="assistant", timestamp="2026-09-03T10:01:00Z", cwd=deep_cwd,
                    message={"role": "assistant", "content": [{"type": "text", "text": "hi"}]}),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-deep", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/Users/tester")))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    assert len(snaps[0].name) <= 200
    data = json.loads(snaps[0].read_text())
    assert deep_cwd.replace("/Users/tester", "~") in data["context"]  # full context kept


def test_claude_code_connector_caps_session_total_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = []
    for i in range(10):
        lines.append(_user_line(f"user turn {i} " + "x" * 500))
        lines.append(_assistant_line([{"type": "text", "text": f"assistant turn {i} " + "y" * 500}]))
    _write_session(projects, "-Users-tester-brain", "sess-big", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setenv("BRAIN_TRANSCRIPT_MAX_CHARS", "2000")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    turns = data["turns"]
    total_chars = sum(len(t["text"]) for t in turns)
    assert total_chars <= 2000
    assert data["stats"]["dropped_overflow_turns"] > 0
    # Both ends kept — the opening turn (never dropped just because a
    # session is long) and the newest turns — with the middle elided via a
    # visible marker turn rather than only ever dropping the oldest turns
    # (review N5).
    assert any("user turn 0 " in t["text"] for t in turns)
    assert any("assistant turn 9" in t["text"] for t in turns)
    assert not any("turn 5 " in t["text"] for t in turns)
    marker_turns = [t for t in turns if t["role"] == "marker"]
    assert len(marker_turns) == 1
    assert "elided" in marker_turns[0]["text"]


def test_claude_code_connector_cap_never_leaves_zero_user_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The head/tail cap (review N5) can still elide every USER turn while
    keeping assistant turns at both ends — the "no real exchange" guard
    must be re-applied AFTER capping, not only before it, so this doesn't
    get written as a one-sided note (or with zero turns at all)."""
    projects = tmp_path / "claude_projects"
    lines = [
        _assistant_line([{"type": "text", "text": "assistant opening " + "a" * 50}]),
        _user_line("user middle " + "u" * 5000),
        _assistant_line([{"type": "text", "text": "assistant closing " + "c" * 5000}]),
    ]
    _write_session(projects, "-Users-tester-brain", "sess-lopsided", lines, mtime=_A_WEEK_AGO)
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))
    monkeypatch.setenv("BRAIN_TRANSCRIPT_MAX_CHARS", "200")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_claude_code_connector_path_flag_beats_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --path (BRAIN_PULL_PATH) must win over BRAIN_CLAUDE_CODE_DIR for this
    # connector too, not just chat_export — it was previously silently
    # ignored here (review N6).
    default_projects = tmp_path / "default_projects"
    lines = [_user_line("from default dir"), _assistant_line([{"type": "text", "text": "ok"}])]
    _write_session(default_projects, "-Users-tester-brain", "sess-default", lines, mtime=_A_WEEK_AGO)

    flag_projects = tmp_path / "flag_projects"
    lines2 = [_user_line("from flag dir"), _assistant_line([{"type": "text", "text": "ok"}])]
    _write_session(flag_projects, "-Users-tester-brain", "sess-flag", lines2, mtime=_A_WEEK_AGO)

    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(default_projects))
    monkeypatch.setenv("BRAIN_PULL_PATH", str(flag_projects))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/claude_code").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["id"] == "sess-flag"


def test_claude_code_connector_path_flag_missing_dir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A configured-but-missing --path is a loud failure, not a silent
    # zero-item pull indistinguishable from "nothing new" (review M2/N6).
    monkeypatch.delenv("BRAIN_CLAUDE_CODE_DIR", raising=False)
    monkeypatch.setenv("BRAIN_PULL_PATH", str(tmp_path / "does-not-exist"))
    paths = _vault(tmp_path)
    with pytest.raises(tc.TranscriptSourceError):
        run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)


def test_claude_code_connector_ignores_subagents_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = tmp_path / "claude_projects"
    lines = [_user_line("hello there"), _assistant_line([{"type": "text", "text": "hi"}])]
    _write_session(projects, "-Users-tester-brain", "sess-main", lines, mtime=_A_WEEK_AGO)
    # A subagents/ sub-folder transcript must not be picked up (non-recursive glob).
    sub = projects / "-Users-tester-brain" / "subagents"
    sub.mkdir(parents=True)
    f = sub / "sess-sub.jsonl"
    f.write_text("".join(lines), encoding="utf-8")
    os.utime(f, (_A_WEEK_AGO, _A_WEEK_AGO))
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(projects))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1


def test_claude_code_connector_explicit_missing_dir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicitly configured (typo'd) dir must be a loud failure, not a
    # silent zero-item pull indistinguishable from "nothing new" (review M2).
    monkeypatch.setenv("BRAIN_CLAUDE_CODE_DIR", str(tmp_path / "does-not-exist"))
    paths = _vault(tmp_path)
    with pytest.raises(tc.TranscriptSourceError):
        run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)


def test_claude_code_connector_missing_default_dir_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The DEFAULT dir being absent (no Claude Code on this machine) is a
    # legitimate "not applicable here", not a misconfiguration — unlike the
    # explicit case above, it must stay silent.
    monkeypatch.delenv("BRAIN_CLAUDE_CODE_DIR", raising=False)
    monkeypatch.setattr(cc, "_DEFAULT_DIR", str(tmp_path / "nobody-home" / "projects"))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["claude_code"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0
