"""Tests for scripts/mcp_probe.py (AUD-123's post-cutover check).

Drives the probe against a real (stubbed) HTTP server on a background
thread — a plain stdlib ``http.server`` speaking just enough of the MCP
Streamable HTTP wire format (a JSON ``initialize`` response with a session
id header, 202 for the ``notifications/initialized`` notification, a
``tools/list`` result) for the probe's hand-rolled client to complete a
real handshake over a real socket. No live server, no network beyond
localhost.
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import ClassVar

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from mcp_probe import (  # noqa: E402
    _parse_token_pair,
    check_health,
    check_initialize_and_tools,
    check_token_status,
    main,
    run_probe,
)


class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Just enough of a brain-mcp server to drive the probe's client:
    GET /health, and POST /mcp for initialize / notifications/initialized /
    tools/list, gated on a Bearer token from a fixed valid set."""

    valid_tokens: ClassVar[frozenset[str]] = frozenset()
    tool_count: ClassVar[int] = 3
    health_status: ClassVar[int] = 200
    # The real server (FastMCP's Streamable HTTP transport, mounted without
    # json_response=True) always replies SSE — that's the default and what
    # every test below exercises unless it opts into json_response mode.
    sse: ClassVar[bool] = True

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # keep test output quiet

    def _write_json(self, status: int, payload: dict[str, object], extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _write_sse(self, status: int, payload: dict[str, object], extra_headers: dict[str, str] | None = None) -> None:
        """Mirror FastMCP's Streamable HTTP wire format: one ``event:
        message`` frame whose ``data:`` line carries the JSON-RPC body."""
        body = f"event: message\r\ndata: {json.dumps(payload)}\r\n\r\n".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _write_jsonrpc(
        self, status: int, payload: dict[str, object], extra_headers: dict[str, str] | None = None
    ) -> None:
        if self.sse:
            self._write_sse(status, payload, extra_headers)
        else:
            self._write_json(status, payload, extra_headers)

    def _write_empty(self, status: int) -> None:
        # Content-Length: 0 (rather than leaving the response unsized) so
        # the client's HTTP/1.1 keep-alive framing stays unambiguous — a
        # sized-but-bodyless response, not a connection the client must
        # read-until-close to know is finished.
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authed(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return auth.removeprefix("Bearer ") in self.valid_tokens

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(self.health_status, {"status": "ok"})
            return
        self._write_empty(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/mcp":
            self._write_empty(404)
            return
        if not self._authed():
            self._write_empty(401)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw)
        method = body.get("method")
        if method == "initialize":
            self._write_jsonrpc(
                200,
                {
                    "jsonrpc": "2.0", "id": body.get("id"),
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "stub-brain-mcp", "version": "0"},
                    },
                },
                extra_headers={"Mcp-Session-Id": "test-session-id"},
            )
        elif method == "notifications/initialized":
            self._write_empty(202)
        elif method == "tools/list":
            tools = [{"name": f"tool_{i}", "inputSchema": {"type": "object"}} for i in range(self.tool_count)]
            self._write_jsonrpc(200, {"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": tools}})
        else:
            self._write_empty(400)


@contextmanager
def _stub_server(
    *, valid_tokens: frozenset[str], tool_count: int = 3, health_status: int = 200, sse: bool = True,
) -> Iterator[str]:
    class Handler(_StubHandler):
        pass

    Handler.valid_tokens = valid_tokens
    Handler.tool_count = tool_count
    Handler.health_status = health_status
    Handler.sse = sse

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _closed_port_url() -> str:
    """A URL nothing is listening on: bind, learn the port, then close it."""
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


# --------------------------------------------------------------- /health


def test_check_health_ok() -> None:
    with _stub_server(valid_tokens=frozenset({"tok"})) as url:
        result = check_health(url)
    assert result.ok is True
    assert result.detail == "200"


def test_check_health_non_200_status_fails() -> None:
    with _stub_server(valid_tokens=frozenset({"tok"}), health_status=503) as url:
        result = check_health(url)
    assert result.ok is False
    assert "503" in result.detail


def test_check_health_connection_refused_fails_cleanly() -> None:
    result = check_health(_closed_port_url())
    assert result.ok is False
    assert "request failed" in result.detail


# --------------------------------------------------------- token auth status


def test_check_token_status_valid_token_is_200() -> None:
    with _stub_server(valid_tokens=frozenset({"good-token"})) as url:
        result = check_token_status(url, "claude-code", "good-token")
    assert result.ok is True
    assert result.name == "token:claude-code"


def test_check_token_status_wrong_token_is_401() -> None:
    with _stub_server(valid_tokens=frozenset({"good-token"})) as url:
        result = check_token_status(url, "codex", "wrong-token")
    assert result.ok is False
    assert "401" in result.detail


# --------------------------------------------------- full MCP handshake


def test_initialize_and_tools_reports_the_real_tool_count() -> None:
    """Happy path against the real wire format: FastMCP's Streamable HTTP
    transport replies SSE (finding #1), so this is the format that must
    parse — a JSON-only stub here would pass without proving anything
    about the live server."""
    with _stub_server(valid_tokens=frozenset({"tok"}), tool_count=18) as url:
        checks, count = check_initialize_and_tools(url, "tok")
    by_name = {c.name: c for c in checks}
    assert by_name["mcp_initialize"].ok is True
    assert by_name["tools_list"].ok is True
    assert count == 18
    assert "18 tools" in by_name["tools_list"].detail


def test_initialize_and_tools_also_works_in_json_response_mode() -> None:
    """A server mounted with json_response=True (not brain-mcp's own
    config, but a valid FastMCP mode) replies plain JSON instead of SSE —
    covered so the content-type branch in _decode_jsonrpc stays correct
    both ways."""
    with _stub_server(valid_tokens=frozenset({"tok"}), tool_count=5, sse=False) as url:
        checks, count = check_initialize_and_tools(url, "tok")
    by_name = {c.name: c for c in checks}
    assert by_name["mcp_initialize"].ok is True
    assert by_name["tools_list"].ok is True
    assert count == 5


def test_initialize_fails_with_wrong_token() -> None:
    with _stub_server(valid_tokens=frozenset({"tok"})) as url:
        checks, count = check_initialize_and_tools(url, "wrong")
    assert len(checks) == 1
    assert checks[0].name == "mcp_initialize"
    assert checks[0].ok is False
    assert count is None
    assert "401" in checks[0].detail


# --------------------------------------------------------------- run_probe


def test_run_probe_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    with _stub_server(valid_tokens=frozenset({"a-token", "b-token"}), tool_count=18) as url:
        report = run_probe(url, {"claude-code": "a-token", "codex": "b-token"})

    assert report.ok is True
    assert report.tool_count == 18
    names = {c.name for c in report.checks}
    assert {"health", "mcp_initialize", "tools_list", "token:claude-code", "token:codex"} <= names


def test_run_probe_one_bad_token_fails_overall_without_hiding_the_rest() -> None:
    with _stub_server(valid_tokens=frozenset({"a-token"})) as url:
        report = run_probe(url, {"claude-code": "a-token", "codex": "wrong-token"})

    assert report.ok is False
    by_name = {c.name: c for c in report.checks}
    assert by_name["health"].ok is True
    assert by_name["token:claude-code"].ok is True
    assert by_name["token:codex"].ok is False


def test_run_probe_requires_at_least_one_token() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_probe("http://127.0.0.1:1", {})


# --------------------------------------------------------------- CLI (main)


def test_main_exits_zero_when_everything_passes(capsys: pytest.CaptureFixture[str]) -> None:
    with _stub_server(valid_tokens=frozenset({"a-token"}), tool_count=18) as url:
        rc = main(["--url", url, "claude-code=a-token"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[ok] health" in out
    assert "18 tool(s)" in out


def test_main_exits_nonzero_on_a_bad_token(capsys: pytest.CaptureFixture[str]) -> None:
    with _stub_server(valid_tokens=frozenset({"a-token"})) as url:
        rc = main(["--url", url, "claude-code=a-token", "codex=wrong-token"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "[FAIL] token:codex" in out


def test_parse_token_pair_accepts_name_equals_token() -> None:
    assert _parse_token_pair("claude-code=abc123") == ("claude-code", "abc123")


def test_parse_token_pair_rejects_missing_equals() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_token_pair("claude-code-abc123")


def test_parse_token_pair_rejects_empty_name_or_token() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_token_pair("=abc123")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_token_pair("claude-code=")
