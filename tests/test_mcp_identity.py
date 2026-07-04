"""Tests for mcp_server.identity (token-spec parsing) and the multi-token
BearerAuthMiddleware, including agent identity propagation via AGENT_VAR.

The middleware is pure ASGI, so we drive it directly with hand-built
scopes and stub receive/send callables — no HTTP server, fully offline.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# mcp_server is not an installed package (only ingest_lib is). The full
# suite imports it via a collection-order side effect; pin the repo root
# onto sys.path so this file also runs standalone.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp_server.auth import BearerAuthMiddleware  # noqa: E402
from mcp_server.identity import current_agent, parse_token_spec  # noqa: E402

TOKEN_A = "a" * 32
TOKEN_B = "b" * 32


# --------------------------------------------------------- parse_token_spec

def test_parse_token_spec_happy_path() -> None:
    spec = f"agent-a={TOKEN_A}, agent_b = {TOKEN_B} ,"  # whitespace + trailing comma
    assert parse_token_spec(spec) == {TOKEN_A: "agent-a", TOKEN_B: "agent_b"}


def test_parse_token_spec_duplicate_name_rejected() -> None:
    with pytest.raises(RuntimeError, match="duplicate agent name"):
        parse_token_spec(f"agent-a={TOKEN_A},agent-a={TOKEN_B}")


def test_parse_token_spec_duplicate_token_rejected() -> None:
    with pytest.raises(RuntimeError, match="reuses one token"):
        parse_token_spec(f"agent-a={TOKEN_A},agent-b={TOKEN_A}")


def test_parse_token_spec_short_token_rejected() -> None:
    with pytest.raises(RuntimeError, match="shorter than 24"):
        parse_token_spec("agent-a=tooshort")


@pytest.mark.parametrize("bad_name", ["Agent-A", "-leading", "_leading", "a" * 33, ""])
def test_parse_token_spec_bad_name_rejected(bad_name: str) -> None:
    with pytest.raises(RuntimeError, match="invalid"):
        parse_token_spec(f"{bad_name}={TOKEN_A}")


def test_parse_token_spec_missing_separator_rejected() -> None:
    with pytest.raises(RuntimeError, match="not name=token"):
        parse_token_spec("justatokenwithoutaname")


def test_parse_token_spec_token_with_equals_rejected() -> None:
    # partition splits on the FIRST '='; the leftover '=' in the token
    # must be refused, not silently kept.
    with pytest.raises(RuntimeError, match="contains '=' or ','"):
        parse_token_spec(f"agent-a={TOKEN_A}=extra")


def test_parse_token_spec_empty_rejected() -> None:
    with pytest.raises(RuntimeError, match="no name=token entries"):
        parse_token_spec(" , ,")


# ------------------------------------------------------------- middleware

def _scope(path: str = "/mcp", token: str | None = None) -> dict:
    headers: list[tuple[bytes, bytes]] = []
    if token is not None:
        headers.append((b"authorization", b"Bearer " + token.encode()))
    return {
        "type": "http",
        "path": path,
        "headers": headers,
        "client": ("127.0.0.1", 12345),
    }


def _drive(scope: dict, **mw_kwargs) -> tuple[dict, list[dict]]:
    """Run one request through a fresh middleware wrapping a stub app.
    Returns (downstream observations, messages sent to the client)."""
    seen: dict = {}
    sent: list[dict] = []

    async def downstream(scope, receive, send):  # noqa: ANN001 — ASGI shape
        # Capture the agent INSIDE the request: AGENT_VAR is reset when
        # the middleware returns, so reading it afterwards would only
        # ever see the default.
        seen["called"] = True
        seen["agent"] = current_agent()

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    mw = BearerAuthMiddleware(downstream, **mw_kwargs)
    asyncio.run(mw(scope, receive, send))
    return seen, sent


def test_valid_token_sets_agent_identity() -> None:
    seen, sent = _drive(
        _scope(token=TOKEN_A),
        tokens={TOKEN_A: "agent-a", TOKEN_B: "agent-b"},
    )
    assert seen == {"called": True, "agent": "agent-a"}
    assert sent == []  # downstream owns the response


def test_second_token_maps_to_its_own_agent() -> None:
    seen, _ = _drive(
        _scope(token=TOKEN_B),
        tokens={TOKEN_A: "agent-a", TOKEN_B: "agent-b"},
    )
    assert seen["agent"] == "agent-b"


def test_invalid_token_gets_401_and_no_downstream() -> None:
    seen, sent = _drive(
        _scope(token="wrong-" + "x" * 26),
        tokens={TOKEN_A: "agent-a"},
    )
    assert seen == {}  # downstream never called
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401


def test_missing_token_gets_401() -> None:
    seen, sent = _drive(_scope(token=None), tokens={TOKEN_A: "agent-a"})
    assert seen == {}
    assert sent[0]["status"] == 401


def test_health_bypasses_auth() -> None:
    seen, sent = _drive(_scope(path="/health", token=None), tokens={TOKEN_A: "agent-a"})
    assert seen["called"] is True
    # No identity on the public path — stays at the fail-safe default.
    assert seen["agent"] == "unknown"
    assert sent == []


def test_middleware_requires_some_token_source() -> None:
    async def noop(scope, receive, send):  # noqa: ANN001
        pass

    with pytest.raises(ValueError):
        BearerAuthMiddleware(noop)


def test_agent_var_reset_after_request() -> None:
    async def noop(scope, receive, send):  # noqa: ANN001
        pass

    async def run() -> str:
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        mw = BearerAuthMiddleware(noop, tokens={TOKEN_A: "agent-a"})
        await mw(_scope(token=TOKEN_A), receive, send)
        # Same task, after the request: identity must not leak.
        return current_agent()

    assert asyncio.run(run()) == "unknown"
