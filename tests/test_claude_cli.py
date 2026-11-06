"""Tests for the locked-down Claude Code CLI runner (ingest_lib.claude_cli).

No test runs a real ``claude``: every run goes through a recorder patched
over the ``claude_cli._run_cli`` seam, fed with the stream-json transcripts in
``tests/fixtures/claude_cli/``. The one exception is the seam's own timeout
test, which runs the real seam on a sleeping Python child.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import JsonValue, TypeAdapter

from ingest_lib import claude_cli
from ingest_lib.claude_cli import CliText, child_env
from ingest_lib.summarize import DocSummary

_FIX = Path(__file__).parent / "fixtures" / "claude_cli"
_LOG = logging.getLogger("test.claude_cli")
_JSON: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_SYSTEM = "You summarise documents."
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
# The real lookup, kept before the autouse fixture replaces it with a stub.
_REAL_CLI_BINARY = claude_cli.cli_binary

_COMMON_FLAGS = [
    "-p", "--model", "sonnet", "--output-format", "stream-json", "--verbose",
    "--setting-sources", "", "--strict-mcp-config", "--disallowedTools", "mcp__*",
    "--disable-slash-commands", "--permission-mode", "dontAsk", "--permission-prompts", "none",
    "--no-session-persistence", "--system-prompt", _SYSTEM, "--settings",
]
_FIXED_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}


def _fixture(name: str) -> str:
    return (_FIX / name).read_text()


def _edit_line(text: str, index: int, **changes: JsonValue) -> str:
    """Return ``text`` with keys of its ``index``-th JSON line replaced."""
    lines = text.splitlines()
    event = _JSON.validate_json(lines[index])
    event.update(changes)
    lines[index] = _JSON.dump_json(event).decode()
    return "\n".join(lines) + "\n"


@dataclass
class Call:
    argv: list[str]
    stdin: str
    cwd: Path
    listing: list[str]
    files: dict[str, bytes]
    env: dict[str, str]


@dataclass
class FakeCli:
    """Recorder patched over ``claude_cli._run_cli``."""

    stdout: Callable[[Path], str] = lambda _cwd: _fixture("text_ok.jsonl")
    returncode: int = 0
    stderr: str = ""
    exc: BaseException | None = None
    calls: list[Call] = field(default_factory=list)

    def use(self, name: str, returncode: int = 0) -> None:
        text = _fixture(name)
        self.stdout = lambda _cwd: text
        self.returncode = returncode

    def __call__(
        self, argv: list[str], *, stdin: str, cwd: Path, env: dict[str, str], timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        listing = sorted(p.name for p in cwd.iterdir())
        files = {n: (cwd / n).read_bytes() for n in listing}
        self.calls.append(Call(argv, stdin, cwd, listing, files, env))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout(cwd), self.stderr)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in ("BRAIN_CLAUDE_CLI_BIN", "BRAIN_CLAUDE_CLI_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(claude_cli, "_blocked", None)
    monkeypatch.setattr(claude_cli, "cli_binary", lambda: "/stub/claude")
    claude_cli._warn_bad_override.cache_clear()
    yield


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch) -> FakeCli:
    fake = FakeCli()
    monkeypatch.setattr(claude_cli, "_run_cli", fake)
    return fake


def _text(user: str = "Document body.") -> CliText | None:
    return claude_cli.run_text(system=_SYSTEM, user=user, model="sonnet", log=_LOG)


def _schema() -> tuple[DocSummary, str] | None:
    return claude_cli.run_structured(system=_SYSTEM, user="Document body.", schema=DocSummary, model="sonnet", log=_LOG)


def _image(suffix: str = ".png") -> CliText | None:
    return claude_cli.run_image(
        image=_PNG, suffix=suffix, system=_SYSTEM, prompt="Describe the image.", model="sonnet", log=_LOG,
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _assert_one_warning(caplog: pytest.LogCaptureFixture, *needles: str) -> None:
    warnings = _warnings(caplog)
    assert len(warnings) == 1, warnings
    assert warnings[0].startswith("claude-code:"), warnings
    for needle in needles:
        assert needle in warnings[0], warnings


def _settings(argv: list[str]) -> dict[str, JsonValue]:
    return _JSON.validate_json(argv[argv.index("--settings") + 1])


# --- argv and profiles ------------------------------------------------------


def test_text_profile_argv_is_the_lockdown(fake_cli: FakeCli) -> None:
    assert _text() is not None
    argv = fake_cli.calls[0].argv
    assert argv == ["/stub/claude", *_COMMON_FLAGS, argv[len(_COMMON_FLAGS) + 1], "--tools", ""]
    assert _settings(argv) == {"disableAllHooks": True}
    for banned in ("--bare", "--safe-mode", "--allowedTools", "--add-dir", "--restricted",
                   "--dangerously-skip-permissions"):
        assert banned not in argv


def test_schema_profile_adds_json_schema(fake_cli: FakeCli) -> None:
    fake_cli.use("schema_ok.jsonl")
    assert _schema() is not None
    argv = fake_cli.calls[0].argv
    assert argv[-4:-1] == ["--tools", "", "--json-schema"]
    assert _JSON.validate_json(argv[-1]) == DocSummary.model_json_schema()
    assert _settings(argv) == {"disableAllHooks": True}


def test_image_profile_confines_read(fake_cli: FakeCli) -> None:
    fake_cli.use("image_ok.jsonl")
    assert _image() is not None
    call = fake_cli.calls[0]
    assert call.argv[-3:] == ["--tools", "Read", "--restricted"]
    assert _settings(call.argv) == {
        "disableAllHooks": True,
        "permissions": {"blockReadsOutsideWorkingDirectories": True},
    }
    assert call.cwd == call.cwd.resolve()
    assert call.listing == ["image.png"]
    assert call.files == {"image.png": _PNG}
    assert call.stdin.startswith("Read ./image.png.")


@pytest.mark.parametrize(
    ("file_path", "accepted"),
    [
        ("./image.png", True),
        ("image.png", True),
        ("{cwd}/image.png", True),
        ("../x/image.png", False),
        ("/tmp/x/image.png", False),
    ],
)
def test_image_read_paths(
    fake_cli: FakeCli, caplog: pytest.LogCaptureFixture, file_path: str, accepted: bool,
) -> None:
    text = _fixture("image_ok.jsonl")
    fake_cli.stdout = lambda cwd: text.replace(
        '"file_path":"./image.png"', '"file_path":"' + file_path.format(cwd=cwd) + '"',
    )
    out = _image()
    if accepted:
        assert out is not None
        assert not _warnings(caplog)
    else:
        assert out is None
        _assert_one_warning(caplog, "Read")


def test_image_read_of_other_file_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("image_read_other.jsonl")
    assert _image() is None
    _assert_one_warning(caplog, "Read")


def test_document_only_on_stdin(fake_cli: FakeCli) -> None:
    nonce = "nonce-5f1c9e2a"
    user = f"A document mentioning {nonce} in its body."
    assert _text(user) is not None
    call = fake_cli.calls[0]
    assert not any(nonce in arg for arg in call.argv)
    assert call.stdin == user


def test_cwd_is_empty_and_removed(fake_cli: FakeCli) -> None:
    assert _text() is not None
    call = fake_cli.calls[0]
    assert call.listing == []
    assert not call.cwd.exists()


# --- child environment ------------------------------------------------------


def test_child_env_is_an_allowlist() -> None:
    base = {
        "ANTHROPIC_API_KEY": "sk-decoy", "BRAIN_MCP_TOKENS": "x", "OPENAI_API_KEY": "y",
        "CLAUDE_CONFIG_DIR": "/c", "HOME": "/h", "PATH": "/p", "HTTPS_PROXY": "http://proxy",
    }
    expected = {"HOME": "/h", "PATH": "/p", "HTTPS_PROXY": "http://proxy", **_FIXED_ENV}
    assert child_env(base) == expected
    del expected["CLAUDE_CODE_DISABLE_CLAUDE_MDS"]
    assert child_env(base, load_claude_mds=True) == expected


def test_run_uses_child_env_of_os_environ(fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-decoy")
    assert _text() is not None
    env = fake_cli.calls[0].env
    assert "ANTHROPIC_API_KEY" not in env
    assert "sk-decoy" not in env.values()
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


# --- successful runs --------------------------------------------------------


def test_text_ok_returns_text_and_model(fake_cli: FakeCli) -> None:
    expected = _JSON.validate_json(_fixture("text_ok.jsonl").splitlines()[-1])["result"]
    assert _text() == CliText(text=str(expected), truncated=False, model_id="claude-sonnet-5-5")


def test_schema_ok_validates(fake_cli: FakeCli) -> None:
    fake_cli.use("schema_ok.jsonl")
    out = _schema()
    assert out is not None
    summary, model_id = out
    assert isinstance(summary, DocSummary)
    assert summary.topics
    # The schema capture ran on Haiku; the id comes from the result's modelUsage.
    assert model_id == "claude-haiku-5-5"


def test_success_without_errors_or_structured_output_parses(fake_cli: FakeCli) -> None:
    result_line = _fixture("text_ok.jsonl").splitlines()[-1]
    assert '"errors"' not in result_line
    assert '"structured_output"' not in result_line
    assert _text() is not None


def test_truncated_from_stop_reason(fake_cli: FakeCli) -> None:
    fake_cli.use("max_tokens.jsonl")
    out = _text()
    assert out is not None
    assert out.truncated


def test_permission_denials_warn_and_return(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = _edit_line(
        _fixture("text_ok.jsonl"), -1,
        permission_denials=[{"tool_name": "Bash", "tool_use_id": "toolu_x", "tool_input": {"command": "ls"}}],
    )
    fake_cli.stdout = lambda _cwd: text
    out = _text()
    assert out is not None
    assert out.text
    _assert_one_warning(caplog, "Bash")


# --- result-level failures --------------------------------------------------


def test_schema_gave_up_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("schema_gave_up.jsonl")
    assert _schema() is None
    _assert_one_warning(caplog, "structured_output")


def test_is_error_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("is_error_model.jsonl", returncode=1)
    assert _text() is None
    _assert_one_warning(caplog, "is_error", "claude-nonexistent-9")


def test_budget_subtype_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("budget.jsonl", returncode=1)
    assert _text() is None
    _assert_one_warning(caplog, "error_max_budget_usd", "Reached maximum budget")


def test_schema_validation_failure_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    lines = _fixture("schema_ok.jsonl").splitlines()
    text = _edit_line("\n".join(lines), len(lines) - 1, structured_output={"summary": "only a summary"})
    fake_cli.stdout = lambda _cwd: text
    assert _schema() is None
    _assert_one_warning(caplog, "DocSummary")


def test_nonzero_exit_empty_stdout_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.stdout = lambda _cwd: ""
    fake_cli.returncode = 2
    fake_cli.stderr = "\n".join(f"line {i}" for i in range(20))
    assert _text() is None
    _assert_one_warning(caplog, "status 2", "line 19", "line 8")
    assert "line 7" not in _warnings(caplog)[0]


def test_line_separator_inside_result_text_parses(fake_cli: FakeCli) -> None:
    result = "first\u2028second\u2029third\x85fourth"
    text = _edit_line(_fixture("text_ok.jsonl"), -1, result=result)
    assert "\u2028" in text
    fake_cli.stdout = lambda _cwd: text
    out = _text()
    assert out is not None
    assert out.text == result


def test_undecodable_output_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    assert _text() is None
    _assert_one_warning(caplog, "UTF-8")


def test_malformed_json_line_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = _fixture("text_ok.jsonl") + "{not json\n"
    fake_cli.stdout = lambda _cwd: text
    assert _text() is None
    _assert_one_warning(caplog, "unreadable")
    assert "not json" not in _warnings(caplog)[0]


# --- init and tool_use invariants -------------------------------------------


def test_init_with_unexpected_tool_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("extra_tool.jsonl")
    assert _text() is None
    _assert_one_warning(caplog, "Bash")


def test_missing_init_event_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = "\n".join(_fixture("text_ok.jsonl").splitlines()[1:]) + "\n"
    fake_cli.stdout = lambda _cwd: text
    assert _text() is None
    _assert_one_warning(caplog, "init")


def test_mcp_server_present_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = _edit_line(_fixture("text_ok.jsonl"), 0, mcp_servers=[{"name": "brain", "status": "connected"}])
    fake_cli.stdout = lambda _cwd: text
    assert _text() is None
    _assert_one_warning(caplog, "brain")


def test_api_key_source_not_none_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = _edit_line(_fixture("text_ok.jsonl"), 0, apiKeySource="ANTHROPIC_API_KEY")
    fake_cli.stdout = lambda _cwd: text
    assert _text() is None
    _assert_one_warning(caplog, "apiKeySource")


def test_unexpected_tool_use_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("unexpected_tool_use.jsonl")
    assert _text() is None
    _assert_one_warning(caplog, "Bash")


def _without_result(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if '"type":"result"' not in line) + "\n"


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda t: _edit_line(t, 0, permissionMode="default"), "permissionMode"),
        (lambda t: _edit_line(t, 0, memory_paths={"auto": "/tmp/x/memory"}), "memory"),
        (_without_result, "result"),
    ],
    ids=["permission_mode", "memory_paths", "no_result_event"],
)
def test_init_invariant_violations_fail_closed(
    fake_cli: FakeCli, caplog: pytest.LogCaptureFixture, mutate: Callable[[str], str], needle: str,
) -> None:
    text = mutate(_fixture("text_ok.jsonl"))
    fake_cli.stdout = lambda _cwd: text
    assert _text() is None
    _assert_one_warning(caplog, needle)


def test_renamed_init_field_fails_closed(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = _fixture("text_ok.jsonl").replace('"permissionMode":', '"permission_mode":', 1)
    fake_cli.stdout = lambda _cwd: text
    assert _text() is None
    _assert_one_warning(caplog, "unreadable")


# --- process-level failures and the breaker ---------------------------------


def test_timeout_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.exc = subprocess.TimeoutExpired(["/stub/claude"], 600)
    assert _text() is None
    _assert_one_warning(caplog, "timed out")
    assert claude_cli.blocked_reason() is None


def test_os_error_returns_none(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.exc = PermissionError(13, "Permission denied")
    assert _text() is None
    _assert_one_warning(caplog, "Permission denied")


def test_cli_binary_valid_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("BRAIN_CLAUDE_CLI_BIN", str(binary))
    monkeypatch.setattr(shutil, "which", lambda _name: "/stub/path/claude")
    assert _REAL_CLI_BINARY() == str(binary)
    assert not _warnings(caplog)


def test_cli_binary_non_executable_override_warns_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o644)
    monkeypatch.setenv("BRAIN_CLAUDE_CLI_BIN", str(binary))
    monkeypatch.setattr(shutil, "which", lambda _name: "/stub/path/claude")
    assert _REAL_CLI_BINARY() == "/stub/path/claude"
    assert _REAL_CLI_BINARY() == "/stub/path/claude"
    _assert_one_warning(caplog, "BRAIN_CLAUDE_CLI_BIN", str(binary))


def test_cli_binary_unset_uses_path(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/stub/path/claude")
    assert _REAL_CLI_BINARY() == "/stub/path/claude"
    assert not _warnings(caplog)


def test_missing_binary_warns_once(
    fake_cli: FakeCli, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_cli, "cli_binary", lambda: None)
    assert _text() is None
    assert _text() is None
    assert fake_cli.calls == []
    _assert_one_warning(caplog, "claude")
    assert claude_cli.blocked_reason()


def test_usage_limit_trips_breaker_without_respawning(
    fake_cli: FakeCli, caplog: pytest.LogCaptureFixture,
) -> None:
    fake_cli.use("usage_limit.jsonl", returncode=1)
    assert _text() is None
    assert _schema() is None
    assert _image() is None
    assert len(fake_cli.calls) == 1
    _assert_one_warning(caplog, "session limit")
    assert claude_cli.blocked_reason()


def test_not_logged_in_trips_breaker(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("not_logged_in.jsonl", returncode=1)
    assert _text() is None
    assert _text() is None
    assert len(fake_cli.calls) == 1
    _assert_one_warning(caplog, "not logged in")
    assert claude_cli.blocked_reason()


def test_not_logged_in_without_init_trips_breaker(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    text = "\n".join(_fixture("not_logged_in.jsonl").splitlines()[1:]) + "\n"
    fake_cli.stdout = lambda _cwd: text
    fake_cli.returncode = 1
    assert _text() is None
    assert _text() is None
    assert len(fake_cli.calls) == 1
    _assert_one_warning(caplog, "not logged in")
    assert claude_cli.blocked_reason()


def test_throttle_text_does_not_trip_breaker(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("throttle.jsonl", returncode=1)
    assert _text() is None
    assert claude_cli.blocked_reason() is None
    _assert_one_warning(caplog, "temporarily limiting")
    caplog.clear()
    assert _text() is None
    assert len(fake_cli.calls) == 2
    _assert_one_warning(caplog, "temporarily limiting")


def test_breaker_ignores_limit_text_in_successful_result(fake_cli: FakeCli) -> None:
    limit_text = "You've hit your usage limit · Not logged in"
    text = _edit_line(_fixture("text_ok.jsonl"), -1, result=limit_text)
    fake_cli.stdout = lambda _cwd: text
    out = _text()
    assert out is not None
    assert out.text == limit_text
    assert claude_cli.blocked_reason() is None


def test_prompt_too_long_returns_none_without_retry(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    fake_cli.use("prompt_too_long.jsonl", returncode=1)
    assert _text("x" * 10_000) is None
    assert len(fake_cli.calls) == 1
    _assert_one_warning(caplog, "prompt_too_long")
    assert claude_cli.blocked_reason() is None


def test_image_rejects_unknown_suffix(fake_cli: FakeCli, caplog: pytest.LogCaptureFixture) -> None:
    assert _image(".heic") is None
    assert fake_cli.calls == []
    _assert_one_warning(caplog, ".heic")


# --- the real seam, on Python, never on claude ------------------------------


# The keyword arguments _run_cli passes to Popen.
_PopenKwarg = str | int | bool | Path | dict[str, str]


def test_seam_decodes_utf8_strictly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, _PopenKwarg] = {}

    def popen(_argv: list[str], **kwargs: _PopenKwarg) -> NoReturn:
        seen.update(kwargs)
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(FileNotFoundError):
        claude_cli._run_cli(["/stub/claude"], stdin="", cwd=tmp_path, env={}, timeout=1)
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "strict"


def test_seam_kills_process_group_on_timeout(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        claude_cli._run_cli(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin="", cwd=tmp_path, env={}, timeout=0.5,
        )
    assert time.monotonic() - started < 5
