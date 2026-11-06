#!/usr/bin/env python3
"""Locked-down launch of the scheduled dream pass (``BRAIN_DREAM_LOCKDOWN=1``).

    scripts/dream_launch.py --max-turns 50 [--claude-json ~/.claude.json]

``scripts/dream.sh`` runs this under its wall-clock cap. It starts ``claude
-p "/dream-pass --scheduled"`` in the vault with only project settings, only
the ``brain`` MCP server (copied from the user's ``~/.claude.json`` into a
private temporary config), ``dontAsk`` permissions, hooks off, no saved
session, an allowlisted environment with no API keys, and Bash confined to
Claude Code's sandbox with the vault's ``.env`` unreadable. The session may
use the six vault tools the skill names, read the vault, write only under
``~/.cache/brain-dream/`` and run ``dream_gate.py``.

The child's stream-json output is copied through unchanged (dream.sh logs
it). Its ``system/init`` event is checked first: any other tool set, MCP
server or server status, permission mode, API key source or loaded memory
stops the run.

Exit codes: the child's own, 2 without a ``brain`` MCP entry (it refuses to
fall back to an unlocked run), 3 on a failed init check, 127 without a
``claude`` binary, 128 + signal number when stopped by SIGTERM or SIGINT.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import FrameType
from typing import Final

# Make ``ingest_lib`` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError  # noqa: E402

from ingest_lib.claude_cli import InitEvent, child_env, cli_binary  # noqa: E402

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

# Credential paths the pass must never read. Kept equal to the legacy copy in
# dream.sh (bash cannot import this), which tests/test_dream_skills.py checks.
DREAM_DENY_RULES: Final[tuple[str, ...]] = (
    "Read(.env)", "Read(.env.*)",
    "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(~/.gnupg/**)",
    "Read(~/.claude/**)", "Read(~/.config/**)",
    "Read(//etc/shadow)", "Read(//etc/ssh/**)",
    "Read(~/.claude.json)", "Read(~/.brain-mcp-token)",
)
_BUILTIN_TOOLS: Final = ("Read", "Grep", "Glob", "Write", "Bash")
_BRAIN_TOOLS: Final = (
    "mcp__brain__vault_read", "mcp__brain__vault_search", "mcp__brain__vault_related",
    "mcp__brain__vault_create_note", "mcp__brain__vault_replace_note",
    "mcp__brain__vault_update_compiled_truth",
)
# Non-secret knobs the session's own dream_gate.py calls read.
_GATE_ENV: Final = ("BRAIN_DREAM_THRESHOLD", "BRAIN_DREAM_STALE_DAYS", "BRAIN_DREAM_PAIRS")
_STOP_WAIT: Final = 30.0
_STOP_SIGNALS: Final = (signal.SIGTERM, signal.SIGINT)


class _Lenient(BaseModel):
    # ~/.claude.json holds many keys this launcher does not use.
    model_config = ConfigDict(extra="ignore")


class _Project(_Lenient):
    mcpServers: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)


class _ClaudeJson(_Lenient):
    mcpServers: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    projects: dict[str, _Project] = Field(default_factory=dict)


_LINE: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])


class _Stop(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def brain_entry(claude_json: Path, repo_root: Path) -> dict[str, JsonValue] | None:
    """The ``brain`` MCP server entry, user scope first, then local scope.

    The entry is opaque config, forwarded to the temporary config verbatim."""
    try:
        config = _ClaudeJson.model_validate_json(claude_json.read_bytes())
    except FileNotFoundError:
        return None
    if "brain" in config.mcpServers:
        return config.mcpServers["brain"]
    project = config.projects.get(str(repo_root))
    return project.mcpServers.get("brain") if project is not None else None


def _settings(repo_root: Path, interpreter_dir: Path, mcp_dir: Path) -> str:
    return json.dumps({
        "disableAllHooks": True,
        "permissions": {"blockReadsOutsideWorkingDirectories": True},
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": False,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                # mcp_dir holds the copied brain entry and its token. The
                # envVars deny below does not hide claude's own environment
                # from /proc on Linux, which the scrub's PID namespace would.
                # The permission layer (dontAsk, reads outside the working
                # folders blocked) is the control relied on; this /proc
                # entry is an extra layer, unverified under bubblewrap.
                "denyRead": [f"{repo_root}/.env", f"{repo_root}/.env.*", str(mcp_dir), "/proc/*/environ"],
                "allowRead": [str(interpreter_dir)],
            },
            # Bash inherits the session's env, so without this a set
            # CLAUDE_CODE_OAUTH_TOKEN reaches every command; claude keeps it.
            # Not CLAUDE_CODE_SUBPROCESS_ENV_SCRUB: on 2.1.318 it forces the
            # permission mode to "default", which the init check rejects.
            "credentials": {"envVars": [{"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"}]},
        },
    })


def build_argv(
    *, binary: str, mcp_config: Path, scratch: Path, repo_root: Path, interpreter_dir: Path, max_turns: int,
) -> list[str]:
    # The positional prompt comes before every variadic flag, which would
    # otherwise swallow it.
    return [
        binary, "-p", "/dream-pass --scheduled",
        "--setting-sources", "project", "--strict-mcp-config", "--mcp-config", str(mcp_config),
        "--tools", ",".join(_BUILTIN_TOOLS), "--add-dir", str(scratch),
        "--allowedTools", *_BRAIN_TOOLS,
        # One leading slash anchors at the settings source; "//" is absolute.
        f"Edit(/{scratch}/**)",
        "Bash(uv run --no-sync python scripts/dream_gate.py *)",
        "Bash(.venv/bin/python scripts/dream_gate.py *)",
        "--disallowedTools", *DREAM_DENY_RULES,
        "--permission-mode", "dontAsk", "--permission-prompts", "none",
        "--no-session-persistence", "--settings", _settings(repo_root, interpreter_dir, mcp_config.parent),
        "--output-format", "stream-json", "--verbose", "--max-turns", str(max_turns),
    ]


def _start(argv: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    # Same process group as the launcher, so dream.sh's timeout (which
    # signals its group, then SIGKILLs it) reaches claude directly.
    return subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, cwd=cwd, env=env)


def _init_failure(init: InitEvent) -> str | None:
    builtins = {t for t in init.tools if not t.startswith("mcp__")}
    if builtins != set(_BUILTIN_TOOLS):
        return f"init built-in tools {sorted(builtins)} differ from the expected {sorted(_BUILTIN_TOOLS)}"
    foreign = sorted(t for t in init.tools if t.startswith("mcp__") and not t.startswith("mcp__brain__"))
    if foreign:
        return f"init lists MCP tools outside the brain server: {foreign}"
    servers = [s.name for s in init.mcp_servers]
    if servers != ["brain"]:
        return f"init lists MCP servers {servers}, not ['brain']"
    if init.mcp_servers[0].status != "connected":
        return f"init brain MCP server status is {init.mcp_servers[0].status!r}, not 'connected'"
    if init.permissionMode != "dontAsk":
        return f"init permissionMode is {init.permissionMode!r}, not 'dontAsk'"
    if init.apiKeySource != "none":
        return f"init apiKeySource is {init.apiKeySource!r}, not 'none'"
    if init.memory_paths is not None:
        return "init loaded memory paths"
    return None


def _line_failure(line: str) -> tuple[bool, str | None]:
    """(decided, failure) for one stdout line, while no init has been seen."""
    if not line.startswith("{"):
        return False, None
    try:
        raw = _LINE.validate_json(line)
        if raw.get("type") != "system":
            return True, "no system/init event before the first session event"
        if raw.get("subtype") != "init":
            return False, None
        return True, _init_failure(InitEvent.model_validate(raw))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(map(str, first["loc"]))
        return True, f"unreadable stream-json event before the init check ({first['type']} at {where})"


def _code(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def _reap(proc: subprocess.Popen[str]) -> int:
    try:
        proc.wait(timeout=_STOP_WAIT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return _code(proc.returncode)


def _supervise(proc: subprocess.Popen[str]) -> int:
    assert proc.stdout is not None  # stdout=PIPE
    checked = False
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if checked:
            continue
        checked, failure = _line_failure(line)
        if failure is not None:
            print(f"[dream_launch] stopping the dream pass: {failure}", file=sys.stderr)
            proc.terminate()
            _reap(proc)
            return 3
    code = _code(proc.wait())
    if not checked:
        print("[dream_launch] the session printed no system/init event", file=sys.stderr)
        return code or 3
    return code


def _raise_stop(signum: int, _frame: FrameType | None) -> None:
    # Ignore further signals before raising, so a second one cannot raise
    # another _Stop while the first unwinds; _run's finally restores them.
    for s in _STOP_SIGNALS:
        signal.signal(s, signal.SIG_IGN)
    raise _Stop(signum)


def _run(argv: list[str], *, env: dict[str, str]) -> int:
    proc: subprocess.Popen[str] | None = None
    previous = {s: signal.signal(s, _raise_stop) for s in _STOP_SIGNALS}
    try:
        proc = _start(argv, cwd=REPO_ROOT, env=env)
        return _supervise(proc)
    except _Stop as stop:
        if proc is None:
            return 128 + stop.signum
        proc.send_signal(stop.signum)
        return _reap(proc)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locked-down launch of the scheduled dream pass.")
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--claude-json", type=Path, default=Path.home() / ".claude.json")
    args = parser.parse_args(argv)
    binary = cli_binary()
    if binary is None:
        print("[dream_launch] no claude binary found (set BRAIN_CLAUDE_CLI_BIN or put claude on PATH)",
              file=sys.stderr)
        return 127
    entry = brain_entry(args.claude_json, REPO_ROOT)
    if entry is None:
        print("[dream_launch] no brain MCP server configured; refusing to run the dream pass unlocked",
              file=sys.stderr)
        return 2
    scratch = Path.home() / ".cache" / "brain-dream"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    scratch.chmod(0o700)  # mkdir leaves an existing folder's mode alone
    # The uv-managed Python the venv links to, so sandboxed Bash can start it.
    interpreter_dir = Path(os.path.realpath(REPO_ROOT / ".venv/bin/python")).parent.parent
    env = child_env(os.environ, load_claude_mds=True)
    env.update({k: os.environ[k] for k in _GATE_ENV if k in os.environ})
    folder = Path(tempfile.mkdtemp(prefix="brain-dream-mcp-")).resolve()
    try:
        config = folder / "mcp.json"
        fd = os.open(config, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"mcpServers": {"brain": entry}}, fh)
        return _run(
            build_argv(
                binary=binary, mcp_config=config, scratch=scratch, repo_root=REPO_ROOT,
                interpreter_dir=interpreter_dir, max_turns=args.max_turns,
            ),
            env=env,
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
