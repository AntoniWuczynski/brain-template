"""Chat-transcript connectors (Claude Code sessions, claude.ai/ChatGPT
conversation exports) + the shared transcript extractor.

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
from ingest_lib.extractors import dispatch_extractor
from ingest_lib.extractors import transcript as transcript_ex

_LOG = logging.getLogger("test")
_AT = "2026-09-03T00:00:00Z"


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


# ------------------------------------------------------------ extractor

def _snapshot(**kw: object) -> bytes:
    return json.dumps(kw).encode("utf-8")


def test_transcript_extractor_renders_note(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(_snapshot(
        connector="claude_code", id="s1", title="brain: fix the thing",
        date="2026-09-03", context="~/brain",
        turns=[
            {"role": "user", "text": "Can you fix the bug?"},
            {"role": "assistant", "text": "Done, here is what changed."},
        ],
        stats={"dropped_tool_uses": 3, "dropped_thinking_blocks": 1},
    ))
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "# brain: fix the thing" in res.markdown
    assert "**Source:** claude_code" in res.markdown
    assert "**Context:** ~/brain" in res.markdown
    assert "Can you fix the bug?" in res.markdown
    assert "Done, here is what changed." in res.markdown
    assert "3 tool calls dropped" in res.markdown
    assert "internal reasoning" in res.markdown


def test_transcript_extractor_no_turns_is_partial(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(_snapshot(connector="chatgpt", id="c1", title="Empty", turns=[]))
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert "no transcript content survived" in res.markdown


def test_transcript_extractor_bad_json_is_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(b"{ not json")
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_transcript_extractor_non_object_is_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(b"[1, 2, 3]")
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_transcript_snapshot_routes_by_source_class() -> None:
    routed = dispatch_extractor(
        Path("x.json"), relative_path="transcripts/claude_code/2026-09-03-x-abcd1234.json"
    )
    assert routed is transcript_ex.extract
    routed2 = dispatch_extractor(
        Path("x.json"), relative_path="transcripts/chat_export/2026-09-03-x-abcd1234.json"
    )
    assert routed2 is transcript_ex.extract


def test_transcripts_registered_in_connectors_and_extractors() -> None:
    assert "claude_code" in CONNECTORS
    assert "chat_export" in CONNECTORS


# ------------------------------------------------------- _transcript_common

@pytest.mark.parametrize(
    "raw,kind",
    [
        ("AKIA" + "A" * 16, "aws-access-key"),
        ("sk-ant-" + "a" * 25, "anthropic-key"),
        ("sk-" + "a" * 25, "openai-key"),
        ("ghp_" + "a" * 40, "github-token"),
        ("xoxb-" + "a" * 15, "slack-token"),
        ("Bearer " + "a" * 25, "bearer-token"),
    ],
)
def test_redact_secrets_known_patterns(raw: str, kind: str) -> None:
    cleaned, n = tc.redact_secrets(f"here is a secret: {raw} end")
    assert n == 1
    assert f"[REDACTED:{kind}]" in cleaned
    assert raw not in cleaned


def test_redact_secrets_private_key_block() -> None:
    block = "-----BEGIN RSA PRIVATE KEY-----\nabc123\n-----END RSA PRIVATE KEY-----"
    cleaned, n = tc.redact_secrets(block)
    assert n == 1
    assert "[REDACTED:private-key-block]" in cleaned


def test_redact_secrets_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dGhpc2lzYXNpZ25hdHVyZQ"
    # Not an env-style "token=..." assignment (which now correctly matches
    # _ENV_ASSIGN_RE first per the C2 fix) — plain prose containing a JWT,
    # to exercise the JWT pattern itself in isolation.
    cleaned, n = tc.redact_secrets(f"here is a raw jwt: {jwt}")
    assert n == 1
    assert "[REDACTED:jwt]" in cleaned


def test_redact_env_style_assignment_keeps_key_name() -> None:
    cleaned, n = tc.redact_secrets('MY_API_KEY="abcdef123456"')
    assert n == 1
    assert "MY_API_KEY=" in cleaned
    assert "[REDACTED:env-secret]" in cleaned
    assert "abcdef123456" not in cleaned


# The refuter's exact table (review C2): a bare canonical keyword — with
# NOTHING before it for a mandatory leading `[A-Za-z_]` to consume — used to
# pass through unredacted entirely.
@pytest.mark.parametrize(
    "raw",
    [
        "TOKEN=supersecretvalue",
        "SECRET=supersecretvalue",
        "PASSWORD=supersecretvalue",
        "APIKEY=supersecretvalue",
        "API_KEY=supersecretvalue",
        "password: supersecretvalue",
    ],
)
def test_redact_env_style_bare_keyword_assignment(raw: str) -> None:
    cleaned, n = tc.redact_secrets(raw)
    assert n == 1
    assert "[REDACTED:env-secret]" in cleaned
    assert "supersecretvalue" not in cleaned


@pytest.mark.parametrize("raw", ["XTOKEN=supersecretvalue", "X_TOKEN=supersecretvalue"])
def test_redact_env_style_prefixed_keyword_still_matches(raw: str) -> None:
    # The prefixed shape already worked before the C2 fix; must keep working.
    cleaned, n = tc.redact_secrets(raw)
    assert n == 1
    assert "[REDACTED:env-secret]" in cleaned


def test_redact_url_credentials() -> None:
    cleaned, n = tc.redact_secrets("DATABASE_URL=postgres://user:hunter2@host/db")
    assert n >= 1
    assert "hunter2" not in cleaned
    assert "[REDACTED:url-credentials]" in cleaned
    assert "postgres://" in cleaned  # scheme kept for context


# The refuter's round-two adversarial table (review N1) — shapes that
# leaked through the round-one redactor untouched because the env-assign
# regex couldn't cross a space/quote/hyphen and had no vendor pattern for
# these prefixes at all.
@pytest.mark.parametrize(
    "raw,secret",
    [
        ("export API_KEY=supersecretvalue", "supersecretvalue"),
        ("export ANTHROPIC_AUTH_TOKEN=supersecretvalue", "supersecretvalue"),
        ("  export SECRET=supersecretvalue", "supersecretvalue"),
        ("env API_KEY=supersecretvalue node app.js", "supersecretvalue"),
        ('{"api_key": "supersecretvalue"}', "supersecretvalue"),
        ('  "apiKey": "supersecretvalue",', "supersecretvalue"),
        ('  "password": "supersecretvalue",', "supersecretvalue"),
        ('"client_secret":"supersecretvalue"', "supersecretvalue"),
        ("x-api-key: supersecretvalue", "supersecretvalue"),
        ("-H 'x-api-key: supersecretvalue'", "supersecretvalue"),
        ("  - token: supersecretvalue", "supersecretvalue"),
        ("https://api.example.com/v1?api_key=supersecretvalue", "supersecretvalue"),
        ("redis://:hunter2pass@127.0.0.1:6379/0", "hunter2pass"),
        ("github_pat_11ABCDEFG0" + "b" * 60, "github_pat_11ABCDEFG0" + "b" * 60),
        ("npm_" + "A" * 36, "npm_" + "A" * 36),
        ("glpat-" + "A" * 20, "glpat-" + "A" * 20),
        ("AIzaSyA1234567890abcdefghijklmnopqrstuv", "AIzaSyA1234567890abcdefghijklmnopqrstuv"),
        ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
    ],
)
def test_redact_secrets_adversarial_table(raw: str, secret: str) -> None:
    cleaned, n = tc.redact_secrets(raw)
    assert n >= 1
    assert "REDACTED" in cleaned
    assert secret not in cleaned


def test_redact_secrets_vendor_pattern_keeps_specific_label_over_generic_one() -> None:
    # A recognised vendor shape sitting right after a generic keyword must
    # keep ITS specific label, not get re-redacted/overwritten by the
    # generic env-secret/http-header net once the vendor pattern already
    # replaced it with a placeholder (review N1 — the double-redaction
    # trap the broadened generic patterns would otherwise fall into).
    cleaned, n = tc.redact_secrets("here is a secret: " + "AKIA" + "A" * 16 + " end")
    assert n == 1
    assert "[REDACTED:aws-access-key]" in cleaned


def test_redact_url_credentials_empty_username() -> None:
    cleaned, n = tc.redact_secrets("redis://:hunter2pass@127.0.0.1:6379/0")
    assert n == 1
    assert "hunter2pass" not in cleaned
    assert "[REDACTED:url-credentials]" in cleaned


def test_redact_basic_auth() -> None:
    cleaned, n = tc.redact_secrets("Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l")
    assert n == 1
    assert "[REDACTED:basic-auth]" in cleaned
    assert "YWxhZGRpbjpvcGVuc2VzYW1l" not in cleaned


def test_home_relative_swaps_prefix() -> None:
    assert tc.home_relative("/Users/alice/brain/scripts", "/Users/alice") == "~/brain/scripts"
    assert tc.home_relative("/Users/alice", "/Users/alice") == "~"
    assert tc.home_relative("/var/other", "/Users/alice") == "/var/other"


def test_finalize_turn_text_strips_wrapper_tags_and_scrubs_home() -> None:
    raw = "<system-reminder>ignore this</system-reminder>Please look at /Users/alice/brain/x.py"
    text, n_redacted, truncated = tc.finalize_turn_text(raw, "/Users/alice")
    assert text is not None
    assert "system-reminder" not in text
    assert "~/brain/x.py" in text
    assert not truncated


def test_finalize_turn_text_collapses_bare_slash_command() -> None:
    raw = "<command-message>dream-pass</command-message>\n<command-name>/dream-pass</command-name>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "_(ran `/dream-pass`)_"


def test_finalize_turn_text_collapses_bare_slash_command_name_first() -> None:
    # The real harness emits command-name FIRST for a typed slash command
    # (review C1) — the command-only match must not be order-dependent.
    raw = (
        "<command-name>/exit</command-name>\n"
        "            <command-message>exit</command-message>\n"
        "            <command-args></command-args>"
    )
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "_(ran `/exit`)_"


def test_finalize_turn_text_collapses_skill_invocation_with_skill_format() -> None:
    # A skill invocation: command-message first, plus a trailing skill-format.
    raw = (
        "<command-message>dream-pass</command-message>\n"
        "<command-name>/dream-pass</command-name>\n"
        "<skill-format>markdown</skill-format>"
    )
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "_(ran `/dream-pass`)_"


def test_finalize_turn_text_returns_none_for_pure_wrapper() -> None:
    raw = "<system-reminder>only harness content</system-reminder>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text is None


@pytest.mark.parametrize(
    "tag",
    [
        "local-command-caveat",
        "task-notification",
        "bash-input",
        "bash-stdout",
        "bash-stderr",
    ],
)
def test_finalize_turn_text_strips_all_observed_wrapper_tags(tag: str) -> None:
    raw = f"<{tag}>harness furniture, not something the user said</{tag}>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text is None, f"{tag} should be fully stripped, leaving nothing"


def test_finalize_turn_text_drops_unrecognised_harness_tag_via_allowlist() -> None:
    # A wrapper tag this module doesn't name yet, but still SHAPED LIKE
    # this harness's own tag vocabulary (a "system-*" tag here) — the
    # allowlist fallback must still drop it rather than keep it verbatim as
    # if it were real prose (review C1, restricted per N2).
    raw = "<system-future-event>new bookkeeping shape</system-future-event>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Not every leading "<word>" is harness — plain prose/markup a
        # developer pasted must survive, even when it happens to open with
        # a tag, as long as that tag isn't shaped like this harness's own
        # wrapper vocabulary (review N2).
        ("<system-reminder>drop me</system-reminder>but keep this part", "but keep this part"),
        (
            "<details>\n<summary>x</summary>\nreal prose\n</details>",
            "<details>\n<summary>x</summary>\nreal prose\n</details>",
        ),
        (
            "<div>  hello world, this is a component I am debugging </div>",
            "<div>  hello world, this is a component I am debugging </div>",
        ),
        (
            "<html><body>Fix this page please</body></html>",
            "<html><body>Fix this page please</body></html>",
        ),
        ("<svg>…</svg>\nwhy does this not render?", "<svg>…</svg>\nwhy does this not render?"),
        (
            "<configuration>…</configuration>\nwhat is wrong here?",
            "<configuration>…</configuration>\nwhat is wrong here?",
        ),
    ],
)
def test_finalize_turn_text_keeps_real_prose_starting_with_angle_bracket_word(
    raw: str, expected: str
) -> None:
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == expected


def test_finalize_turn_text_keeps_prose_after_a_leading_command_name_tag() -> None:
    # A <command-name> element mixed into otherwise-real prose (not a PURE
    # bare-command wrapper) must have just that element stripped, not make
    # the allowlist net mistake the remainder for an unrecognised wrapper
    # and drop the whole turn (review N2).
    raw = "<command-name>/foo</command-name>\nand here is my actual question about the build"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "and here is my actual question about the build"


def test_finalize_turn_text_truncates_long_turns() -> None:
    raw = "x" * (tc._MAX_TURN_CHARS + 500)
    text, _n, truncated = tc.finalize_turn_text(raw, "")
    assert truncated is True
    assert text is not None
    assert len(text) < len(raw)
    assert "truncated" in text


# ------------------------------------------------------------ claude_code

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


# ------------------------------------------------------------ chat_export

def test_chat_export_connector_claude_ai_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {
            "uuid": "conv-1", "name": "Planning a trip",
            "created_at": "2026-09-01T10:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "Where should I go?"},
                {"sender": "assistant", "content": [
                    {"type": "text", "text": "How about Kyoto?"},
                    {"type": "image", "source": {}},
                ]},
                {"sender": "system", "text": "irrelevant bookkeeping"},
            ],
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["connector"] == "claude_ai"
    assert data["id"] == "conv-1"
    assert data["date"] == "2026-09-01"
    turns = data["turns"]
    assert turns == [
        {"role": "user", "text": "Where should I go?"},
        {"role": "assistant", "text": "How about Kyoto?"},
    ]
    assert data["stats"]["dropped_non_text_parts"] == 1
    assert data["stats"]["dropped_other_events"] == 1  # the "system" sender


def test_chat_export_connector_chatgpt_shape_orders_by_create_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"

    def node(author_role: str, text: str, create_time: float) -> dict[str, object]:
        return {
            "message": {
                "author": {"role": author_role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": create_time,
            }
        }

    export.write_text(json.dumps([
        {
            "id": "conv-gpt-1", "title": "Debugging session",
            "create_time": 1735689600,
            "mapping": {
                "n2": node("assistant", "Second: here's the fix.", 200),
                "n1": node("user", "First: it crashes on start.", 100),
                "n3": node("system", "hidden", 50),
                "n4": {"message": {"author": {"role": "user"},
                                   "content": {"content_type": "code", "parts": ["x=1"]}}},
            },
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["connector"] == "chatgpt"
    assert data["turns"] == [
        {"role": "user", "text": "First: it crashes on start."},
        {"role": "assistant", "text": "Second: here's the fix."},
    ]
    assert data["stats"]["dropped_other_events"] == 1     # system author
    assert data["stats"]["dropped_non_text_parts"] == 1   # non-text content_type


def test_chat_export_connector_chatgpt_shape_includes_all_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented behaviour (module docstring): every node's message is
    walked, ordered by create_time — NOT just the current_node path — so an
    edited-and-regenerated reply is interleaved into the transcript rather
    than silently dropped. This was previously untested (review m4); it is
    a deliberate choice, not a bug, so this test only pins the documented
    shape rather than changing it."""
    export = tmp_path / "conversations.json"

    def node(author_role: str, text: str, create_time: float) -> dict[str, object]:
        return {
            "message": {
                "author": {"role": author_role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": create_time,
            }
        }

    export.write_text(json.dumps([
        {
            "id": "conv-fork-1", "title": "Edited prompt",
            "create_time": 1735689600,
            "mapping": {
                "n1": node("user", "How do I center a div?", 100),
                # Two ALTERNATIVE assistant replies to the same prompt —
                # one before the user edited/regenerated, one after.
                "n2a": node("assistant", "Use margin: auto.", 200),
                "n2b": node("assistant", "Use flexbox: display: flex.", 300),
            },
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    # Both alternative branches survive, ordered by create_time.
    assert data["turns"] == [
        {"role": "user", "text": "How do I center a div?"},
        {"role": "assistant", "text": "Use margin: auto."},
        {"role": "assistant", "text": "Use flexbox: display: flex."},
    ]


def test_chat_export_connector_missing_id_fallback_is_stable_across_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation with none of uuid/conversation_id/id must get the SAME
    native id whether it's first or second in the export list — a
    position-based fallback breaks idempotency the moment an export is
    re-run with items reordered/prepended (review m5)."""
    conv = {"name": "No id here", "created_at": "2026-09-01T00:00:00Z",
            "chat_messages": [{"sender": "human", "text": "hi"},
                              {"sender": "assistant", "text": "yo"}]}
    other = {"uuid": "conv-other", "name": "Other", "created_at": "2026-08-01T00:00:00Z",
              "chat_messages": [{"sender": "human", "text": "hi"},
                               {"sender": "assistant", "text": "yo"}]}

    export_a = tmp_path / "a.json"
    export_a.write_text(json.dumps([conv, other]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export_a))
    paths_a = _vault(tmp_path)
    run_connector(CONNECTORS["chat_export"](), paths_a, pulled_at=_AT, logger=_LOG)
    ids_a = {json.loads(f.read_text())["id"]
             for f in (paths_a.root / "inbox/transcripts/chat_export").glob("*.json")}

    export_b = tmp_path / "b.json"
    export_b.write_text(json.dumps([other, conv]), encoding="utf-8")  # reordered
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export_b))
    paths_b = paths_for_root(tmp_path / "vault-b")
    paths_b.ensure()
    run_connector(CONNECTORS["chat_export"](), paths_b, pulled_at=_AT, logger=_LOG)
    ids_b = {json.loads(f.read_text())["id"]
             for f in (paths_b.root / "inbox/transcripts/chat_export").glob("*.json")}

    assert ids_a == ids_b


def test_chat_export_connector_unrecognized_shape_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([{"id": "x", "some_other_shape": True}]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_chat_export_connector_no_env_yields_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAIN_CHAT_EXPORT_PATH", raising=False)
    monkeypatch.delenv("BRAIN_PULL_PATH", raising=False)
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_chat_export_connector_falls_back_to_generic_pull_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-1", "name": "T", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"},
                          {"sender": "assistant", "text": "hello"}]},
    ]), encoding="utf-8")
    monkeypatch.delenv("BRAIN_CHAT_EXPORT_PATH", raising=False)
    monkeypatch.setenv("BRAIN_PULL_PATH", str(export))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1


def test_chat_export_connector_explicit_path_flag_beats_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A .env-supplied BRAIN_CHAT_EXPORT_PATH must not silently win over an
    # explicit --path (BRAIN_PULL_PATH) — the more specific, more recently
    # expressed configuration (review M2).
    default_export = tmp_path / "env_default.json"
    default_export.write_text(json.dumps([
        {"uuid": "conv-env", "name": "From env", "created_at": "2026-01-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "yo"}]},
    ]), encoding="utf-8")
    flag_export = tmp_path / "cli_flag.json"
    flag_export.write_text(json.dumps([
        {"uuid": "conv-flag", "name": "From flag", "created_at": "2026-01-02T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "yo"}]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(default_export))
    monkeypatch.setenv("BRAIN_PULL_PATH", str(flag_export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["id"] == "conv-flag"


def test_chat_export_connector_missing_configured_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A configured-but-missing path is a loud failure, not a silent
    # zero-item pull indistinguishable from "nothing new" (review M2).
    monkeypatch.delenv("BRAIN_CHAT_EXPORT_PATH", raising=False)
    monkeypatch.setenv("BRAIN_PULL_PATH", str(tmp_path / "typo.json"))
    paths = _vault(tmp_path)
    with pytest.raises(tc.TranscriptSourceError):
        run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)


def test_chat_export_connector_long_title_does_not_break_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_title = "Refactor the pipeline " * 20  # ~460 chars, user-controlled
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-long", "name": long_title, "created_at": "2026-08-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "yo"}]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    assert len(snaps) == 1
    assert len(snaps[0].name) <= 200
    data = json.loads(snaps[0].read_text())
    assert data["title"] == long_title  # full title kept in the note itself


def test_chat_export_connector_caps_conversation_total_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    messages = []
    for i in range(10):
        messages.append({"sender": "human", "text": f"user turn {i} " + "x" * 500})
        messages.append({"sender": "assistant", "text": f"assistant turn {i} " + "y" * 500})
    export.write_text(json.dumps([
        {"uuid": "conv-big", "name": "Big", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": messages},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    monkeypatch.setenv("BRAIN_TRANSCRIPT_MAX_CHARS", "2000")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    total_chars = sum(len(t["text"]) for t in data["turns"])
    assert total_chars <= 2000
    assert data["stats"]["dropped_overflow_turns"] > 0
    assert any("assistant turn 9" in t["text"] for t in data["turns"])


def test_chat_export_connector_respects_max_and_orders_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    convs = []
    for i in range(3):
        convs.append({
            "uuid": f"conv-{i}", "name": f"Conversation {i}",
            "created_at": f"2026-09-0{i + 1}T00:00:00Z",
            "chat_messages": [{"sender": "human", "text": "hi"},
                              {"sender": "assistant", "text": "hello"}],
        })
    export.write_text(json.dumps(convs), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_MAX", "1")
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["id"] == "conv-2"   # newest created_at kept


def test_chat_export_connector_claude_ai_non_text_only_message_not_double_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A message whose only content is a non-text part is already counted
    # via dropped_non_text_parts — it must not ALSO be counted as a dropped
    # "wrapper turn" under a second, misleading label (review N4).
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {
            "uuid": "conv-imgonly", "name": "Image only reply",
            "created_at": "2026-09-01T00:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "hi"},
                {"sender": "assistant", "content": [{"type": "image", "source": {}}]},
                {"sender": "assistant", "text": "real reply"},
            ],
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["stats"]["dropped_non_text_parts"] == 1
    assert data["stats"]["dropped_wrapper_turns"] == 0


def test_chat_export_connector_chatgpt_non_string_part_only_not_double_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"

    def node(author_role: str, content: dict[str, object], create_time: float) -> dict[str, object]:
        return {"message": {"author": {"role": author_role}, "content": content,
                             "create_time": create_time}}

    export.write_text(json.dumps([
        {
            "id": "conv-gpt-nontext", "title": "Non-string part",
            "mapping": {
                "n1": node("user", {"content_type": "text", "parts": ["hi"]}, 100),
                "n2": node("assistant", {"content_type": "text", "parts": [{"weird": "struct"}]}, 150),
                "n3": node("assistant", {"content_type": "text", "parts": ["real reply"]}, 200),
            },
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["stats"]["dropped_non_text_parts"] == 1
    assert data["stats"]["dropped_wrapper_turns"] == 0


def test_chat_export_connector_cap_never_leaves_zero_user_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-lopsided", "name": "Lopsided", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": [
             {"sender": "assistant", "text": "assistant opening " + "a" * 50},
             {"sender": "human", "text": "user middle " + "u" * 5000},
             {"sender": "assistant", "text": "assistant closing " + "c" * 5000},
         ]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    monkeypatch.setenv("BRAIN_TRANSCRIPT_MAX_CHARS", "200")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_chat_export_connector_drops_conversation_with_no_kept_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-empty", "name": "Empty", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": [{"sender": "system", "text": "just bookkeeping"}]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


# ------------------------------------------------------------------ pull.py

def test_pull_path_flag_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    _scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import pull

    monkeypatch.delenv("BRAIN_PULL_PATH", raising=False)
    pull.main(["--list", "--path", "/tmp/some/export.json"])
    assert os.environ.get("BRAIN_PULL_PATH") == "/tmp/some/export.json"
