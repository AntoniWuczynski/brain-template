"""Tests for the locked-down dream-pass launcher (scripts/dream_launch.py).

No test runs a real ``claude``. ``main`` starts the child through the
``dream_launch._start`` seam, which these tests replace with a recorder that
starts a short Python child printing a canned stream-json transcript.
"""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import dream_launch  # noqa: E402

_BIN = "/tmp/x/claude"
_ENTRY: dict[str, JsonValue] = {
    "type": "http", "url": "http://127.0.0.1:8765/mcp", "headers": {"Authorization": "Bearer t0ken"},
}
_SESSION = "00000000-0000-4000-8000-000000000000"
_BRAIN_TOOLS = [
    "mcp__brain__vault_read", "mcp__brain__vault_search", "mcp__brain__vault_related",
    "mcp__brain__vault_create_note", "mcp__brain__vault_replace_note",
    "mcp__brain__vault_update_compiled_truth",
]
_SETTINGS: dict[str, JsonValue] = {
    "disableAllHooks": True,
    "permissions": {"blockReadsOutsideWorkingDirectories": True},
    "sandbox": {
        "enabled": True,
        "failIfUnavailable": True,
        "autoAllowBashIfSandboxed": False,
        "allowUnsandboxedCommands": False,
        "filesystem": {
            "denyRead": ["/tmp/x/vault/.env", "/tmp/x/vault/.env.*", "/tmp/x/cfg", "/proc/*/environ"],
            "allowRead": ["/tmp/x/python"],
        },
        "credentials": {"envVars": [{"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"}]},
    },
}


def _init(**changes: JsonValue) -> str:
    event: dict[str, JsonValue] = {
        "type": "system", "subtype": "init", "cwd": "/tmp/x", "session_id": _SESSION,
        "tools": ["Read", "Grep", "Glob", "Write", "Bash", *_BRAIN_TOOLS],
        "mcp_servers": [{"name": "brain", "status": "connected"}],
        "model": "claude-sonnet-5-5", "permissionMode": "dontAsk", "apiKeySource": "none",
        "claude_code_version": "2.1.318",
    }
    event.update(changes)
    return json.dumps(event)


_ASSISTANT = json.dumps({
    "type": "assistant", "session_id": _SESSION,
    "message": {"content": [{"type": "text", "text": "dreaming"}]},
})
_RESULT = json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": _SESSION})


def _write_claude_json(path: Path, data: dict[str, JsonValue]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@dataclass
class FakeStart:
    """Recorder over ``dream_launch._start``: starts a Python child that
    prints ``lines`` and then sleeps ``linger`` seconds."""

    lines: list[str]
    linger: float = 0.0
    exit_code: int = 0
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    config_mode: int = 0
    config_text: str = ""

    def __call__(self, argv: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
        self.argv, self.cwd, self.env = argv, cwd, env
        config = Path(argv[argv.index("--mcp-config") + 1])
        self.config_mode = stat.S_IMODE(config.stat().st_mode)
        self.config_text = config.read_text(encoding="utf-8")
        payload = "".join(f"{ln}\n" for ln in self.lines)
        script = (
            "import sys, time\n"
            f"sys.stdout.write({json.dumps(payload)})\n"
            "sys.stdout.flush()\n"
            f"time.sleep({self.linger})\n"
            f"sys.exit({self.exit_code})\n"
        )
        return subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(dream_launch, "cli_binary", lambda: _BIN)
    return tmp_path


def _claude_json(home: Path) -> Path:
    return _write_claude_json(home / ".claude.json", {"mcpServers": {"brain": _ENTRY}})


def _main(home: Path, fake: FakeStart, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(dream_launch, "_start", fake)
    return dream_launch.main(["--max-turns", "7", "--claude-json", str(_claude_json(home))])


def test_build_argv_is_the_dream_lockdown() -> None:
    scratch = Path("/tmp/x/.cache/brain-dream")
    argv = dream_launch.build_argv(
        binary=_BIN, mcp_config=Path("/tmp/x/cfg/mcp.json"), scratch=scratch,
        repo_root=Path("/tmp/x/vault"), interpreter_dir=Path("/tmp/x/python"), max_turns=50,
    )
    assert argv == [
        _BIN, "-p", "/dream-pass --scheduled",
        "--setting-sources", "project", "--strict-mcp-config", "--mcp-config", "/tmp/x/cfg/mcp.json",
        "--tools", "Read,Grep,Glob,Write,Bash", "--add-dir", "/tmp/x/.cache/brain-dream",
        "--allowedTools", *_BRAIN_TOOLS,
        "Edit(//tmp/x/.cache/brain-dream/**)",
        "Bash(uv run --no-sync python scripts/dream_gate.py *)",
        "Bash(.venv/bin/python scripts/dream_gate.py *)",
        "--disallowedTools",
        "Read(.env)", "Read(.env.*)", "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(~/.gnupg/**)",
        "Read(~/.claude/**)", "Read(~/.config/**)", "Read(//etc/shadow)", "Read(//etc/ssh/**)",
        "Read(~/.claude.json)", "Read(~/.brain-mcp-token)",
        "--permission-mode", "dontAsk", "--permission-prompts", "none",
        "--no-session-persistence", "--settings", json.dumps(_SETTINGS),
        "--output-format", "stream-json", "--verbose", "--max-turns", "50",
    ]
    assert [a for a in argv if a.startswith("Edit(")] == ["Edit(//tmp/x/.cache/brain-dream/**)"]
    assert not any(a.startswith("Write(") for a in argv)
    assert "--disable-slash-commands" not in argv
    assert "--bare" not in argv


def test_settings_sandbox_denies_env() -> None:
    argv = dream_launch.build_argv(
        binary=_BIN, mcp_config=Path("/tmp/x/cfg/mcp.json"), scratch=Path("/tmp/x/.cache/brain-dream"),
        repo_root=Path("/tmp/x/vault"), interpreter_dir=Path("/tmp/x/python"), max_turns=50,
    )
    settings = json.loads(argv[argv.index("--settings") + 1])
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] is True
    assert sandbox["failIfUnavailable"] is True
    assert sandbox["autoAllowBashIfSandboxed"] is False
    assert sandbox["allowUnsandboxedCommands"] is False
    # The temporary MCP config folder holds the brain token, and on Linux
    # /proc/<claude pid>/environ still holds CLAUDE_CODE_OAUTH_TOKEN.
    assert sandbox["filesystem"]["denyRead"] == [
        "/tmp/x/vault/.env", "/tmp/x/vault/.env.*", "/tmp/x/cfg", "/proc/*/environ",
    ]
    assert sandbox["credentials"] == {"envVars": [{"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"}]}
    assert settings["disableAllHooks"] is True


def test_brain_entry_user_scope(tmp_path: Path) -> None:
    path = _write_claude_json(tmp_path / "c.json", {
        "mcpServers": {"brain": _ENTRY, "other": {"command": "x"}},
        "projects": {"/tmp/x/vault": {"mcpServers": {"brain": {"command": "local"}}}},
        "numStartups": 4,
    })
    assert dream_launch.brain_entry(path, Path("/tmp/x/vault")) == _ENTRY


def test_brain_entry_local_scope(tmp_path: Path) -> None:
    path = _write_claude_json(tmp_path / "c.json", {
        "mcpServers": {"other": {"command": "x"}},
        "projects": {
            "/tmp/x/elsewhere": {"mcpServers": {"brain": {"command": "wrong"}}},
            "/tmp/x/vault": {"mcpServers": {"brain": _ENTRY}, "allowedTools": []},
        },
    })
    assert dream_launch.brain_entry(path, Path("/tmp/x/vault")) == _ENTRY


def test_brain_entry_missing_returns_none(tmp_path: Path) -> None:
    path = _write_claude_json(tmp_path / "c.json", {
        "mcpServers": {"other": {"command": "x"}},
        "projects": {"/tmp/x/elsewhere": {"mcpServers": {"brain": _ENTRY}}},
    })
    assert dream_launch.brain_entry(path, Path("/tmp/x/vault")) is None
    assert dream_launch.brain_entry(tmp_path / "absent.json", Path("/tmp/x/vault")) is None


def test_main_refuses_without_brain_entry(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeStart([_init()])
    monkeypatch.setattr(dream_launch, "_start", fake)
    path = _write_claude_json(home / ".claude.json", {"mcpServers": {}})
    assert dream_launch.main(["--max-turns", "7", "--claude-json", str(path)]) == 2
    assert fake.argv == []
    assert "no brain MCP server configured; refusing to run the dream pass unlocked" in capsys.readouterr().err


def test_main_without_binary_exits_127(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeStart([_init()])
    monkeypatch.setattr(dream_launch, "cli_binary", lambda: None)
    assert _main(home, fake, monkeypatch) == 127
    assert fake.argv == []


def test_main_writes_0600_config_and_removes_it(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeStart([_init(), _RESULT])
    assert _main(home, fake, monkeypatch) == 0
    assert fake.config_mode == 0o600
    assert json.loads(fake.config_text) == {"mcpServers": {"brain": _ENTRY}}
    config = Path(fake.argv[fake.argv.index("--mcp-config") + 1])
    # The sandbox denyRead entry must name the real path (macOS /var is a link).
    assert str(config.parent) == os.path.realpath(config.parent)
    assert not config.exists()
    assert not config.parent.exists()
    scratch = home / ".cache/brain-dream"
    assert fake.argv[fake.argv.index("--add-dir") + 1] == str(scratch)
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
    assert fake.cwd == dream_launch.REPO_ROOT
    assert fake.argv[fake.argv.index("--max-turns") + 1] == "7"


def test_main_tightens_existing_scratch_dir(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = home / ".cache/brain-dream"
    scratch.mkdir(parents=True)
    scratch.chmod(0o755)
    assert _main(home, FakeStart([_init(), _RESULT]), monkeypatch) == 0
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700


def test_main_child_env_has_no_secrets(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-decoy-0000")
    monkeypatch.setenv("BRAIN_MCP_TOKENS", "decoy=0000")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "1")
    monkeypatch.setenv("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "1")
    monkeypatch.setenv("BRAIN_DREAM_THRESHOLD", "3")
    monkeypatch.setenv("BRAIN_DREAM_PAIRS", "4")
    monkeypatch.delenv("BRAIN_DREAM_STALE_DAYS", raising=False)
    fake = FakeStart([_init(), _RESULT])
    assert _main(home, fake, monkeypatch) == 0
    assert "ANTHROPIC_API_KEY" not in fake.env
    assert "BRAIN_MCP_TOKENS" not in fake.env
    assert "CLAUDE_CODE_DISABLE_CLAUDE_MDS" not in fake.env
    assert fake.env["BRAIN_DREAM_THRESHOLD"] == "3"
    assert fake.env["BRAIN_DREAM_PAIRS"] == "4"
    assert "BRAIN_DREAM_STALE_DAYS" not in fake.env
    assert fake.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    # The scrub forces permissionMode "default" (live, CLI 2.1.318), which
    # fails the init check; sandbox.credentials keeps the token from Bash.
    assert "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB" not in fake.env
    assert fake.env["HOME"] == str(home)


@pytest.mark.parametrize(
    "init",
    [
        _init(tools=["Read", "Grep", "Glob", "Write", "Bash", "Edit", *_BRAIN_TOOLS]),
        _init(tools=["Read", "Grep", "Glob", "Write", "Bash", *_BRAIN_TOOLS, "mcp__slack__send"]),
        _init(mcp_servers=[{"name": "brain", "status": "connected"}, {"name": "slack", "status": "connected"}]),
        _init(mcp_servers=[{"name": "brain", "status": "failed", "source": "dynamic"}]),
        _init(mcp_servers=[{"name": "brain"}]),
        _init(permissionMode="auto"),
        _init(apiKeySource="ANTHROPIC_API_KEY"),
        _init(memory_paths={"auto": "/tmp/x/memory/"}),
        _init(permissionMode=None),
        _ASSISTANT,
    ],
    ids=[
        "extra-builtin", "foreign-mcp-tool", "second-mcp-server", "brain-not-connected",
        "brain-without-status", "auto-mode", "api-key", "memory",
        "renamed-field", "no-init",
    ],
)
def test_main_bad_init_exits_3(
    init: str, home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeStart([init, _ASSISTANT, _RESULT], linger=20)
    started = time.monotonic()
    assert _main(home, fake, monkeypatch) == 3
    assert time.monotonic() - started < 10, "the child was not terminated"
    captured = capsys.readouterr()
    assert _RESULT not in captured.out
    assert len(captured.err.strip().splitlines()) == 1
    assert not Path(fake.argv[fake.argv.index("--mcp-config") + 1]).exists()


def test_main_good_init_passes_output_through(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    lines = [
        json.dumps({"type": "system", "subtype": "status", "session_id": _SESSION}),
        _init(), _ASSISTANT, _RESULT,
    ]
    fake = FakeStart(lines, exit_code=5)
    assert _main(home, fake, monkeypatch) == 5
    assert capsys.readouterr().out == "".join(f"{ln}\n" for ln in lines)


def test_main_does_not_start_a_new_session(tmp_path: Path) -> None:
    """GNU timeout signals its own process group, so the child must stay in
    the launcher's group and session for the TERM and the later KILL to
    reach it. The real seam runs a sleeping Python child here, never claude."""
    script = "import sys, time; print('ready', flush=True); time.sleep(20)"
    proc = dream_launch._start(
        [sys.executable, "-c", script], cwd=tmp_path, env={"PATH": os.environ["PATH"]},
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline() == "ready\n"
        assert os.getpgid(proc.pid) == os.getpgrp()
        assert os.getsid(proc.pid) == os.getsid(0)
        assert proc.stderr is None
    finally:
        proc.kill()
        proc.wait()
        assert proc.stdout is not None
        proc.stdout.close()


def test_sigterm_is_forwarded_and_config_removed(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStart([_init()], linger=20)
    started: list[subprocess.Popen[str]] = []

    def start(argv: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
        started.append(fake(argv, cwd=cwd, env=env))
        threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
        return started[0]

    monkeypatch.setattr(dream_launch, "_start", start)
    handler = signal.getsignal(signal.SIGTERM)
    began = time.monotonic()
    code = dream_launch.main(["--max-turns", "7", "--claude-json", str(_claude_json(home))])
    assert time.monotonic() - began < 10
    assert started[0].returncode == -signal.SIGTERM
    assert code == 128 + signal.SIGTERM
    assert not Path(fake.argv[fake.argv.index("--mcp-config") + 1]).exists()
    assert signal.getsignal(signal.SIGTERM) == handler


def test_stop_handler_ignores_further_signals_before_raising() -> None:
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(dream_launch._Stop):
            dream_launch._raise_stop(signal.SIGTERM, None)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


# The child signals the launcher itself, so neither signal can arrive before
# its SIGTERM handler is installed: the first once it is, the second while
# the launcher waits for it to exit after the forwarded SIGTERM.
_SLOW_EXIT = """
import os, signal, sys, time
stopping = []
signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
print(INIT, flush=True)
os.kill(os.getppid(), signal.SIGTERM)
while not stopping:
    time.sleep(0.05)
time.sleep(0.5)
os.kill(os.getppid(), signal.SIGTERM)
time.sleep(2.5)
sys.exit(7)
"""


def test_second_signal_while_reaping_is_ignored(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second SIGTERM while the launcher waits for the child must not
    escape as an uncaught _Stop. The stub child takes 3 s to exit after the
    forwarded SIGTERM and sends the second signal inside that wait."""
    started: list[subprocess.Popen[str]] = []

    def start(argv: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
        script = f"INIT = {_init()!r}\n{_SLOW_EXIT}"
        started.append(subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True))
        return started[0]

    monkeypatch.setattr(dream_launch, "_start", start)
    handler = signal.getsignal(signal.SIGTERM)
    code = dream_launch.main(["--max-turns", "7", "--claude-json", str(_claude_json(home))])
    assert code == 7
    assert started[0].returncode == 7
    assert signal.getsignal(signal.SIGTERM) == handler
