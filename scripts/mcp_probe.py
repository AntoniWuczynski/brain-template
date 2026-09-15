#!/usr/bin/env python3
"""Post-cutover smoke test for the brain MCP server (AUD-123).

Checks, over the network, exactly what mcp/DEPLOY.md's post-cutover step
asks an operator to confirm after moving the server: is it actually up,
does the MCP protocol work end to end, and does every configured client
token still authenticate. Meant to be run once right after cutover (or a
token rotation) and on any "is the server OK" doubt afterwards.

    uv run --no-sync python scripts/mcp_probe.py \\
        --url https://brain.example.com \\
        claude-code=<token> codex=<token> claude-ai=<token>

Exits 0 only if every check passed. Each check is printed with its own
ok/FAIL and detail (an HTTP status or error message), so a single bad
token among several doesn't hide behind an overall failure.

Deliberately hand-rolls the JSON-RPC exchange with the stdlib
(``urllib.request``) rather than the ``mcp`` package's client — this is a
handful of plain HTTP POSTs, one per check, and stdlib-only keeps the probe
runnable from a bare Python without a project venv (e.g. straight from the
runbook, before ``uv sync`` has even happened on a fresh box).
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final, cast

_MCP_PATH: Final[str] = "/mcp"
_HEALTH_PATH: Final[str] = "/health"
_PROTOCOL_VERSION: Final[str] = "2025-06-18"
_TIMEOUT_S: Final[float] = 10.0
_SESSION_ID_HEADER: Final[str] = "mcp-session-id"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ProbeReport:
    checks: tuple[CheckResult, ...]
    tool_count: int | None

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _post_json(
    url: str, payload: dict[str, object], *, token: str, session_id: str | None = None
) -> tuple[int, dict[str, str], bytes]:
    """POST one JSON-RPC message. Never raises for a non-2xx status or a
    connection failure — those come back as ``(0, {}, b"<error>")`` for the
    caller to report as a failed check rather than an unhandled traceback."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {token}",
    }
    if session_id is not None:
        headers[_SESSION_ID_HEADER] = session_id
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        resp_headers = dict(exc.headers.items()) if exc.headers is not None else {}
        return exc.code, {k.lower(): v for k, v in resp_headers.items()}, exc.read()
    except OSError as exc:
        return 0, {}, str(exc).encode("utf-8")


def _decode_jsonrpc(headers: dict[str, str], body: bytes) -> dict[str, object]:
    """Decode one JSON-RPC message from an MCP response body.

    FastMCP's Streamable HTTP transport replies with SSE
    (``text/event-stream``) unless the app is mounted with
    ``json_response=True`` — brain-mcp's is not (``mcp_server/app.py``
    calls ``mcp.streamable_http_app()`` with no override), so every real
    response is SSE: one ``event: message`` frame whose ``data:`` line
    carries the JSON-RPC payload. Take that line in the SSE case;
    otherwise (a ``json_response=True`` server, or a plain-JSON stub)
    parse the whole body.

    Raises ``json.JSONDecodeError`` on malformed or missing JSON, and lets
    a ``UnicodeDecodeError`` from decoding a non-UTF-8 SSE body propagate.
    """
    content_type = headers.get("content-type", "")
    if content_type.startswith("text/event-stream"):
        text = body.decode("utf-8")
        for line in text.splitlines():
            if line.startswith("data:"):
                decoded = json.loads(line.removeprefix("data:").strip())
                return cast(dict[str, object], decoded)
        raise json.JSONDecodeError("no data: line in SSE body", text, 0)
    decoded = json.loads(body)
    return cast(dict[str, object], decoded)


def _initialize_payload() -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mcp_probe", "version": "1.0"},
        },
    }


def check_health(base_url: str) -> CheckResult:
    url = base_url.rstrip("/") + _HEALTH_PATH
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except OSError as exc:
        return CheckResult("health", False, f"request failed: {exc}")
    if status == 200:
        return CheckResult("health", True, "200")
    return CheckResult("health", False, f"status {status}")


def check_token_status(base_url: str, name: str, token: str) -> CheckResult:
    """One bare ``initialize`` call authenticated as ``token``: a 200 means
    it authenticates; a 401/403 means it's wrong, unrotated, or missing
    server-side."""
    url = base_url.rstrip("/") + _MCP_PATH
    status, _headers, body = _post_json(url, _initialize_payload(), token=token)
    if status == 200:
        return CheckResult(f"token:{name}", True, "200")
    if status == 0:
        return CheckResult(f"token:{name}", False, f"request failed: {body.decode('utf-8', 'replace')}")
    return CheckResult(f"token:{name}", False, f"status {status}")


def check_initialize_and_tools(base_url: str, token: str) -> tuple[list[CheckResult], int | None]:
    """One full handshake — ``initialize`` -> ``notifications/initialized``
    -> ``tools/list`` — using ``token``. Always returns an ``mcp_initialize``
    check; a ``tools_list`` check is added too once initialize succeeds, so
    the report shows exactly which stage failed rather than one check whose
    name changes depending on where the handshake broke. Returns the tool
    count alongside a passing ``tools_list``, ``None`` otherwise."""
    checks: list[CheckResult] = []
    url = base_url.rstrip("/") + _MCP_PATH
    status, headers, body = _post_json(url, _initialize_payload(), token=token)
    if status != 200:
        detail = f"status {status}" if status else f"request failed: {body.decode('utf-8', 'replace')}"
        checks.append(CheckResult("mcp_initialize", False, detail))
        return checks, None
    session_id = headers.get(_SESSION_ID_HEADER)
    try:
        init_result = _decode_jsonrpc(headers, body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        checks.append(CheckResult("mcp_initialize", False, f"bad JSON: {exc}"))
        return checks, None
    if "error" in init_result:
        checks.append(CheckResult("mcp_initialize", False, f"server error: {init_result['error']}"))
        return checks, None
    checks.append(CheckResult("mcp_initialize", True, "200"))

    notif: dict[str, object] = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    _post_json(url, notif, token=token, session_id=session_id)

    list_payload: dict[str, object] = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    status, list_headers, body = _post_json(url, list_payload, token=token, session_id=session_id)
    if status != 200:
        detail = f"status {status}" if status else f"request failed: {body.decode('utf-8', 'replace')}"
        checks.append(CheckResult("tools_list", False, detail))
        return checks, None
    try:
        list_result = _decode_jsonrpc(list_headers, body)
        result_field = cast(dict[str, object], list_result["result"])
        tools = result_field["tools"]
        if not isinstance(tools, list):
            raise TypeError("'tools' is not a list")
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        checks.append(CheckResult("tools_list", False, f"unexpected response shape: {exc}"))
        return checks, None
    checks.append(CheckResult("tools_list", True, f"{len(tools)} tools"))
    return checks, len(tools)


def run_probe(base_url: str, tokens: dict[str, str]) -> ProbeReport:
    if not tokens:
        raise ValueError("at least one name=token pair is required")
    checks = [check_health(base_url)]
    first_name = next(iter(tokens))
    mcp_checks, tool_count = check_initialize_and_tools(base_url, tokens[first_name])
    checks.extend(mcp_checks)
    for name, token in tokens.items():
        checks.append(check_token_status(base_url, name, token))
    return ProbeReport(checks=tuple(checks), tool_count=tool_count)


def _parse_token_pair(raw: str) -> tuple[str, str]:
    name, sep, token = raw.partition("=")
    if not sep or not name or not token:
        raise argparse.ArgumentTypeError(f"expected name=token, got {raw!r}")
    return name, token


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-probe",
        description=(
            "Post-cutover smoke test for the brain MCP server: GET /health, "
            "one MCP initialize + tools/list handshake, and every given "
            "token's auth status. Exit code is 0 only if every check passed."
        ),
    )
    parser.add_argument("--url", required=True, help="Base URL, e.g. https://brain.example.com")
    parser.add_argument(
        "tokens", nargs="+", type=_parse_token_pair, metavar="name=token",
        help="One or more agent name=token pairs, e.g. claude-code=abc123...",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    url: str = args.url
    pairs: list[tuple[str, str]] = args.tokens
    tokens = dict(pairs)

    report = run_probe(url, tokens)
    for check in report.checks:
        status = "ok" if check.ok else "FAIL"
        print(f"mcp-probe: [{status}] {check.name}: {check.detail}")
    if report.tool_count is not None:
        print(f"mcp-probe: server exposes {report.tool_count} tool(s)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
