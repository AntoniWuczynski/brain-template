"""Locked-down runner for the local Claude Code CLI (``claude -p``).

The ``claude-code`` provider runs the LLM features on a Claude subscription
through the local Claude Code CLI, with no API key. Every run is headless and
locked down: no settings files, hooks, MCP servers, slash commands, saved
session or memory, ``dontAsk`` permissions, an allowlisted environment, an
empty temporary working folder, and the document on stdin only. There are
three profiles. ``text`` and ``schema`` give the model no tools at all (the
schema profile adds ``--json-schema``, which the CLI answers through its own
``StructuredOutput`` tool). ``image`` gives it only ``Read``, confined to the
one image copied into the working folder. The ``stream-json`` transcript is
checked before any answer is used, and any drift (a new tool in the init
event, a tool call outside the profile, an MCP server, an API key in use)
fails closed with one warning. The flag set was verified against Claude Code
2.1.318 and its documentation.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

PROVIDER: Final[str] = "claude-code"
DEFAULT_MODEL: Final[str] = "sonnet"

_COPIED_ENV: Final = (
    "HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LANG",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "CLAUDE_CODE_OAUTH_TOKEN",
)
_FIXED_ENV: Final[Mapping[str, str]] = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
_IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
_LIMIT_RE: Final = re.compile(r"hit your .{0,40}limit", re.I)
_STDERR_TAIL: Final = 12
_LOG: Final = logging.getLogger(__name__)

# Breaker: set once a missing binary, a lost login or a plan limit makes every
# further call pointless. While set, every run_* returns None at once.
_blocked: str | None = None


@dataclass(frozen=True)
class CliText:
    text: str
    truncated: bool
    model_id: str


class _Lenient(BaseModel):
    # The CLI emits many undocumented keys, so unknown keys are ignored.
    model_config = ConfigDict(extra="ignore")


class McpServer(_Lenient):
    name: str
    status: str | None = None


class InitEvent(_Lenient):
    tools: list[str]
    mcp_servers: list[McpServer]
    permissionMode: str
    apiKeySource: str
    model: str
    claude_code_version: str = ""
    memory_paths: dict[str, str] | None = None


class _Event(_Lenient):
    type: str
    subtype: str | None = None


class _Message(_Lenient):
    content: list[dict[str, JsonValue]]


class _Assistant(_Lenient):
    message: _Message


class _ToolUse(_Lenient):
    name: str
    input: dict[str, JsonValue]


class _Denial(_Lenient):
    tool_name: str


class _ModelUsage(_Lenient):
    pass


class _Result(_Lenient):
    subtype: str
    is_error: bool
    stop_reason: str | None
    permission_denials: list[_Denial]
    modelUsage: dict[str, _ModelUsage]
    result: str | None = None
    errors: list[str] | None = None
    structured_output: dict[str, JsonValue] | None = None
    terminal_reason: str | None = None
    api_error_status: int | None = None

    @property
    def failed(self) -> bool:
        return self.is_error or self.subtype != "success"


_LINE: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True)
class _Profile:
    tail: tuple[str, ...]
    settings: str
    tools: frozenset[str]


@dataclass(frozen=True)
class _Stream:
    inits: list[InitEvent]
    tool_uses: list[_ToolUse]
    result: _Result | None


@dataclass(frozen=True)
class _Run:
    result: _Result
    model_id: str


_HOOKS_OFF: Final = json.dumps({"disableAllHooks": True})
_TEXT: Final = _Profile(("--tools", ""), _HOOKS_OFF, frozenset())
_IMAGE: Final = _Profile(
    ("--tools", "Read", "--restricted"),
    json.dumps({"disableAllHooks": True, "permissions": {"blockReadsOutsideWorkingDirectories": True}}),
    frozenset({"Read"}),
)


@functools.cache
def _warn_bad_override(path: str) -> None:
    # Once per path, not once per document.
    _LOG.warning(
        "claude-code: BRAIN_CLAUDE_CLI_BIN=%s is not an executable file; falling back to claude on PATH", path,
    )


def cli_binary() -> str | None:
    override = os.environ.get("BRAIN_CLAUDE_CLI_BIN")
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        _warn_bad_override(override)
    return shutil.which("claude")


def child_env(base: Mapping[str, str], *, load_claude_mds: bool = False) -> dict[str, str]:
    env = {k: base[k] for k in _COPIED_ENV if k in base}
    env.update({k: v for k, v in base.items() if k.startswith("LC_")})
    env.update(_FIXED_ENV)
    if not load_claude_mds:
        env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    return env


def blocked_reason() -> str | None:
    return _blocked


def _run_cli(
    argv: list[str], *, stdin: str, cwd: Path, env: dict[str, str], timeout: float,
) -> subprocess.CompletedProcess[str]:
    out: str
    err: str
    with subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="strict", cwd=cwd, env=env, start_new_session=True,
    ) as proc:
        try:
            out, err = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _trip(log: logging.Logger, reason: str) -> None:
    global _blocked
    _blocked = reason
    log.warning("claude-code: %s; skipping every further claude-code call in this run", reason)


def _breaker_reason(result: _Result) -> str | None:
    for text in (result.result or "", *(result.errors or [])):
        if "not logged in" in text.lower():
            return "not logged in to Claude Code (run `claude` once and log in)"
        if _LIMIT_RE.search(text):
            return f"Claude plan limit reached ({text[:120]})"
    return None


def _parse_stream(stdout: str) -> _Stream:
    inits: list[InitEvent] = []
    tool_uses: list[_ToolUse] = []
    result: _Result | None = None
    # split("\n"), not splitlines(): JSON strings may hold a raw U+2028,
    # U+2029 or U+0085, which splitlines() treats as line breaks.
    for line in stdout.split("\n"):
        if not line.startswith("{"):
            continue
        raw = _LINE.validate_json(line)
        event = _Event.model_validate(raw)
        if event.type == "system" and event.subtype == "init":
            inits.append(InitEvent.model_validate(raw))
        elif event.type == "assistant":
            for block in _Assistant.model_validate(raw).message.content:
                if block.get("type") == "tool_use":
                    tool_uses.append(_ToolUse.model_validate(block))
        elif event.type == "result":
            result = _Result.model_validate(raw)
    return _Stream(inits, tool_uses, result)


def _tool_use_ok(use: _ToolUse, tools: frozenset[str], read_paths: frozenset[str]) -> bool:
    if use.name not in tools:
        return False
    if use.name != "Read":
        return True
    file_path = use.input.get("file_path")
    return isinstance(file_path, str) and file_path in read_paths


def _verdict(stream: _Stream, returncode: int, tools: frozenset[str], read_paths: frozenset[str]) -> _Result | str:
    """The run's result event, or why the run must be discarded."""
    result = stream.result
    if result is not None and result.failed:
        detail = result.result or "; ".join(result.errors or [])
        return (
            f"run failed (subtype={result.subtype}, is_error={result.is_error}, "
            f"terminal_reason={result.terminal_reason}): {detail[:200]}"
        )
    if returncode != 0:
        return f"exited with status {returncode}"
    if not stream.inits:
        return "no system/init event in the output"
    for init in stream.inits:
        if set(init.tools) != tools:
            return f"init tools {sorted(init.tools)} differ from the expected {sorted(tools)}"
        if init.mcp_servers:
            return f"init lists MCP servers {[s.name for s in init.mcp_servers]}"
        if init.permissionMode != "dontAsk":
            return f"init permissionMode is {init.permissionMode!r}, not 'dontAsk'"
        if init.apiKeySource != "none":
            return f"init apiKeySource is {init.apiKeySource!r}, not 'none'"
        if init.memory_paths is not None:
            return "init loaded memory paths"
    for use in stream.tool_uses:
        if not _tool_use_ok(use, tools, read_paths):
            return f"unexpected {use.name} tool use"
    if result is None:
        return "no result event in the output"
    return result


def _check(
    proc: subprocess.CompletedProcess[str], profile: _Profile, read_paths: frozenset[str], log: logging.Logger,
) -> _Run | None:
    try:
        stream = _parse_stream(proc.stdout)
    except ValidationError as exc:
        first = exc.errors()[0]
        log.warning(
            "claude-code: unreadable stream-json event (%s at %s)", first["type"], ".".join(map(str, first["loc"])),
        )
        return None
    result = stream.result
    if result is not None and (result.failed or proc.returncode != 0):
        tripped = _breaker_reason(result)
        if tripped is not None:
            _trip(log, tripped)
            return None
    verdict = _verdict(stream, proc.returncode, profile.tools, read_paths)
    if isinstance(verdict, str):
        reason = verdict
        if proc.returncode != 0 and proc.stderr:
            tail = "\n".join(proc.stderr.splitlines()[-_STDERR_TAIL:])
            reason = f"{reason}; stderr tail:\n{tail}"
        log.warning("claude-code: %s", reason)
        return None
    # _verdict accepts a run only after checking it has an init event.
    model_id = next(iter(verdict.modelUsage), stream.inits[0].model)
    log.debug("claude-code: CLI %s, model %s", stream.inits[0].claude_code_version, model_id)
    if verdict.permission_denials:
        names = sorted({d.tool_name for d in verdict.permission_denials})
        log.warning("claude-code: the model attempted denied tool calls %s (result kept)", names)
    return _Run(verdict, model_id)


def _invoke(
    profile: _Profile, *, system: str, stdin: str, model: str, log: logging.Logger,
    image: tuple[str, bytes] | None = None,
) -> _Run | None:
    if _blocked is not None:
        return None
    binary = cli_binary()
    if binary is None:
        _trip(log, "no claude binary found (set BRAIN_CLAUDE_CLI_BIN or put claude on PATH)")
        return None
    argv = [
        binary, "-p", "--model", model, "--output-format", "stream-json", "--verbose",
        "--setting-sources", "", "--strict-mcp-config", "--disallowedTools", "mcp__*",
        "--disable-slash-commands", "--permission-mode", "dontAsk", "--permission-prompts", "none",
        "--no-session-persistence", "--system-prompt", system, "--settings", profile.settings,
        *profile.tail,
    ]
    timeout = float(os.environ.get("BRAIN_CLAUDE_CLI_TIMEOUT") or 600)
    with tempfile.TemporaryDirectory(prefix="brain-claude-cli-") as td:
        cwd = Path(td).resolve()
        read_paths: frozenset[str] = frozenset()
        if image is not None:
            name, data = image
            (cwd / name).write_bytes(data)
            read_paths = frozenset({f"./{name}", name, str(cwd / name)})
        try:
            proc = _run_cli(argv, stdin=stdin, cwd=cwd, env=child_env(os.environ), timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("claude-code: timed out after %ss (BRAIN_CLAUDE_CLI_TIMEOUT)", timeout)
            return None
        except OSError as exc:
            log.warning("claude-code: could not run %s (%s)", binary, exc.strerror or type(exc).__name__)
            return None
        except UnicodeDecodeError:
            log.warning("claude-code: the CLI's output is not valid UTF-8")
            return None
        return _check(proc, profile, read_paths, log)


def _text_of(run: _Run | None, log: logging.Logger) -> CliText | None:
    if run is None:
        return None
    if run.result.result is None:
        log.warning("claude-code: successful run returned no result text")
        return None
    return CliText(run.result.result, run.result.stop_reason == "max_tokens", run.model_id)


def run_text(*, system: str, user: str, model: str, log: logging.Logger) -> CliText | None:
    return _text_of(_invoke(_TEXT, system=system, stdin=user, model=model, log=log), log)


def run_structured[M: BaseModel](
    *, system: str, user: str, schema: type[M], model: str, log: logging.Logger,
) -> tuple[M, str] | None:
    profile = _Profile(
        ("--tools", "", "--json-schema", json.dumps(schema.model_json_schema())),
        _HOOKS_OFF,
        frozenset({"StructuredOutput"}),
    )
    run = _invoke(profile, system=system, stdin=user, model=model, log=log)
    if run is None:
        return None
    if run.result.structured_output is None:
        log.warning("claude-code: schema run returned no structured_output")
        return None
    try:
        parsed = schema.model_validate(run.result.structured_output)
    except ValidationError as exc:
        log.warning(
            "claude-code: structured_output failed %s validation (%d errors)", schema.__name__, exc.error_count(),
        )
        return None
    return parsed, run.model_id


def run_image(
    *, image: bytes, suffix: str, system: str, prompt: str, model: str, log: logging.Logger,
) -> CliText | None:
    if suffix not in _IMAGE_SUFFIXES:
        log.warning("claude-code: unsupported image type %r", suffix)
        return None
    run = _invoke(
        _IMAGE, system=system, stdin=f"Read ./image{suffix}. {prompt}", model=model, log=log,
        image=(f"image{suffix}", image),
    )
    return _text_of(run, log)
