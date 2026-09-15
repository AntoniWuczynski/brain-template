"""Tests for mcp_server.identity (token-spec parsing) and the multi-token
BearerAuthMiddleware, including agent identity propagation via AGENT_VAR.

The middleware is pure ASGI, so we drive it directly with hand-built
scopes and stub receive/send callables — no HTTP server, fully offline.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest
from starlette.types import Message as _Message
from starlette.types import Receive as _Receive
from starlette.types import Scope as _Scope
from starlette.types import Send as _Send

# mcp_server is not an installed package (only ingest_lib is). The full
# suite imports it via a collection-order side effect; pin the repo root
# onto sys.path so this file also runs standalone.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp_server import auth as auth_mod  # noqa: E402
from mcp_server.auth import BearerAuthMiddleware, _SessionOwners  # noqa: E402
from mcp_server.config import MAX_REQUEST_BYTES  # noqa: E402
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

def _scope(path: str = "/mcp", token: str | None = None) -> _Scope:
    headers: list[tuple[bytes, bytes]] = []
    if token is not None:
        headers.append((b"authorization", b"Bearer " + token.encode()))
    return {
        "type": "http",
        "path": path,
        "headers": headers,
        "client": ("127.0.0.1", 12345),
    }


def _drive(scope: _Scope, *, tokens: dict[str, str]) -> tuple[_Message, list[_Message]]:
    """Run one request through a fresh middleware wrapping a stub app.
    Returns (downstream observations, messages sent to the client)."""
    seen: _Message = {}
    sent: list[_Message] = []

    async def downstream(scope: _Scope, receive: _Receive, send: _Send) -> None:
        # Capture the agent INSIDE the request: AGENT_VAR is reset when
        # the middleware returns, so reading it afterwards would only
        # ever see the default.
        seen["called"] = True
        seen["agent"] = current_agent()

    async def receive() -> _Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: _Message) -> None:
        sent.append(message)

    mw = BearerAuthMiddleware(downstream, tokens=tokens)
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
    async def noop(scope: _Scope, receive: _Receive, send: _Send) -> None:
        pass

    with pytest.raises(ValueError, match=r"non-empty tokens"):
        BearerAuthMiddleware(noop)


def test_agent_var_reset_after_request() -> None:
    async def noop(scope: _Scope, receive: _Receive, send: _Send) -> None:
        pass

    async def run() -> str:
        async def receive() -> _Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: _Message) -> None:
            pass

        mw = BearerAuthMiddleware(noop, tokens={TOKEN_A: "agent-a"})
        await mw(_scope(token=TOKEN_A), receive, send)
        # Same task, after the request: identity must not leak.
        return current_agent()

    assert asyncio.run(run()) == "unknown"


# ------------------------------------------------------- session ownership
# The Streamable-HTTP transport routes by ``mcp-session-id``, so the token
# check alone let any valid token drive a session another agent opened.

SESSION_ID = "f" * 32
OTHER_SESSION_ID = "e" * 32


class _Stub:
    """Stub ASGI app: records the identity it ran under and optionally
    answers with an ``mcp-session-id`` header, as the transport does when
    it mints a session on ``initialize``."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.mint: str | None = None

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        self.calls.append(current_agent())
        headers = [(b"content-type", b"application/json")]
        if self.mint is not None:
            headers.append((b"mcp-session-id", self.mint.encode()))
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": b"{}"})


def _session_scope(token: str, session_id: str, method: str = "POST") -> _Scope:
    scope = _scope(token=token)
    scope["method"] = method
    scope["headers"].append((b"mcp-session-id", session_id.encode()))
    return scope


def _run(mw: BearerAuthMiddleware, scope: _Scope) -> list[_Message]:
    """Drive one request through an EXISTING middleware instance — session
    ownership is per-instance state, so the same instance must serve them all."""
    sent: list[_Message] = []

    async def receive() -> _Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: _Message) -> None:
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    return sent


def _two_agent_mw(stub: _Stub) -> BearerAuthMiddleware:
    return BearerAuthMiddleware(stub, tokens={TOKEN_A: "agent-a", TOKEN_B: "agent-b"})


def test_minted_session_is_bound_to_the_agent_that_opened_it() -> None:
    stub = _Stub()
    mw = _two_agent_mw(stub)
    stub.mint = SESSION_ID
    opened = _run(mw, _scope(token=TOKEN_A))  # initialize: no session header yet
    stub.mint = None

    # The response is forwarded untouched (the claiming send only looks).
    assert opened[0]["status"] == 200
    assert (b"mcp-session-id", SESSION_ID.encode()) in opened[0]["headers"]
    assert opened[1] == {"type": "http.response.body", "body": b"{}"}

    sent = _run(mw, _session_scope(TOKEN_B, SESSION_ID))
    assert stub.calls == ["agent-a"]  # agent-b never reached the app
    assert sent[0]["status"] == 403


def test_session_owner_may_keep_using_its_own_session() -> None:
    stub = _Stub()
    mw = _two_agent_mw(stub)
    stub.mint = SESSION_ID
    _run(mw, _scope(token=TOKEN_A))
    stub.mint = None

    sent = _run(mw, _session_scope(TOKEN_A, SESSION_ID))
    assert stub.calls == ["agent-a", "agent-a"]
    assert sent[0]["status"] == 200


def test_session_first_seen_on_a_request_binds_to_that_agent() -> None:
    stub = _Stub()
    mw = _two_agent_mw(stub)
    assert _run(mw, _session_scope(TOKEN_B, SESSION_ID))[0]["status"] == 200
    assert _run(mw, _session_scope(TOKEN_A, SESSION_ID))[0]["status"] == 403
    assert stub.calls == ["agent-b"]


def test_request_without_session_id_is_unaffected() -> None:
    stub = _Stub()
    mw = _two_agent_mw(stub)
    assert _run(mw, _scope(token=TOKEN_A))[0]["status"] == 200
    assert _run(mw, _scope(token=TOKEN_B))[0]["status"] == 200
    assert stub.calls == ["agent-a", "agent-b"]


def test_two_agents_hold_their_own_sessions() -> None:
    stub = _Stub()
    mw = _two_agent_mw(stub)
    assert _run(mw, _session_scope(TOKEN_A, SESSION_ID))[0]["status"] == 200
    assert _run(mw, _session_scope(TOKEN_B, OTHER_SESSION_ID))[0]["status"] == 200
    assert stub.calls == ["agent-a", "agent-b"]


def test_delete_releases_the_binding() -> None:
    stub = _Stub()
    mw = _two_agent_mw(stub)
    _run(mw, _session_scope(TOKEN_A, SESSION_ID))
    _run(mw, _session_scope(TOKEN_A, SESSION_ID, method="DELETE"))
    # Session terminated by its owner: the id is free again.
    assert _run(mw, _session_scope(TOKEN_B, SESSION_ID))[0]["status"] == 200
    assert stub.calls == ["agent-a", "agent-a", "agent-b"]


def test_session_owner_map_is_bounded() -> None:
    owners = _SessionOwners(limit=2)
    assert owners.claim("s1", "agent-a") == "agent-a"
    assert owners.claim("s2", "agent-a") == "agent-a"
    assert owners.claim("s3", "agent-a") == "agent-a"
    # Oldest binding evicted, so s1 is claimable again; s2/s3 still held.
    assert owners.claim("s1", "agent-b") == "agent-b"
    assert owners.claim("s3", "agent-b") == "agent-a"


def test_concurrent_claims_agree_on_one_owner() -> None:
    owners = _SessionOwners()
    results: list[str] = []
    lock = threading.Lock()

    def claim(i: int) -> None:
        owner = owners.claim(SESSION_ID, f"agent-{i}")
        with lock:
            results.append(owner)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 32
    assert len(set(results)) == 1


# ------------------------------------------------------ request-body limits
# AUD-054: MAX_REQUEST_BYTES bounds ONE body, but the per-tool concurrency
# guards only engage once a request has been fully received and parsed, so
# concurrent max-size bodies were bounded by nothing.


class _BodyApp:
    """Stub ASGI app that drains the request body and can be held in flight
    (so a second request overlaps the first) before it answers."""

    def __init__(self) -> None:
        self.hold = asyncio.Event()
        self.started = asyncio.Event()
        self.received: list[int] = []   # body bytes delivered, per request
        self.disconnects = 0

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        total = 0
        cut = False
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                self.disconnects += 1
                cut = True
                break
            total += len(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        self.received.append(total)
        self.started.set()
        if not cut:
            # A cut-off request must not be held: it is the one whose caller
            # is still waiting to release the hold.
            await self.hold.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})


async def _drive_body(
    mw: BearerAuthMiddleware, chunks: list[int], *, declare: bool = True
) -> list[_Message]:
    """One authenticated request whose body arrives as ``chunks`` of bytes.
    ``declare=False`` omits Content-Length (chunked transfer)."""
    scope = _scope(token=TOKEN_A)
    if declare:
        scope["headers"].append((b"content-length", str(sum(chunks)).encode()))
    remaining = list(chunks)
    sent: list[_Message] = []

    async def receive() -> _Message:
        if not remaining:
            return {"type": "http.request", "body": b"", "more_body": False}
        size = remaining.pop(0)
        return {"type": "http.request", "body": b"x" * size, "more_body": bool(remaining)}

    async def send(message: _Message) -> None:
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def test_declared_body_that_does_not_fit_the_inflight_budget_is_refused() -> None:
    app = _BodyApp()
    mw = BearerAuthMiddleware(
        app, tokens={TOKEN_A: "agent-a"}, max_inflight_body_bytes=1000,
    )

    async def scenario() -> list[_Message]:
        first = asyncio.create_task(_drive_body(mw, [600]))
        await app.started.wait()
        second = await _drive_body(mw, [600])   # 600 + 600 > 1000
        app.hold.set()
        await first
        return second

    sent = asyncio.run(scenario())
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503
    assert app.received == [600]   # the second body never reached the app


def test_chunked_body_is_cut_off_when_the_inflight_budget_is_exhausted() -> None:
    app = _BodyApp()
    mw = BearerAuthMiddleware(
        app, tokens={TOKEN_A: "agent-a"}, max_inflight_body_bytes=1000,
    )

    async def scenario() -> None:
        first = asyncio.create_task(_drive_body(mw, [600]))
        await app.started.wait()
        # No Content-Length: nothing to check up front, so the cap has to
        # bite as the chunks arrive.
        await _drive_body(mw, [400, 400], declare=False)
        app.hold.set()
        await first

    asyncio.run(scenario())
    assert app.disconnects == 1
    assert app.received == [600, 400]   # second request cut after one chunk


def test_inflight_budget_is_released_when_a_request_finishes() -> None:
    app = _BodyApp()
    app.hold.set()   # never hold: each request completes before the next
    mw = BearerAuthMiddleware(
        app, tokens={TOKEN_A: "agent-a"}, max_inflight_body_bytes=1000,
    )

    async def scenario() -> None:
        for _ in range(3):
            await _drive_body(mw, [600])

    asyncio.run(scenario())
    assert app.received == [600, 600, 600]


def test_declared_body_over_the_per_request_cap_gets_413() -> None:
    app = _BodyApp()
    app.hold.set()
    mw = BearerAuthMiddleware(app, tokens={TOKEN_A: "agent-a"})
    scope = _scope(token=TOKEN_A)
    scope["headers"].append((b"content-length", str(MAX_REQUEST_BYTES + 1).encode()))
    sent: list[_Message] = []

    async def receive() -> _Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: _Message) -> None:
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    assert sent[0]["status"] == 413
    assert app.received == []


# AUD-075: the per-request cap has two enforcement points and only one of
# them was covered. A DECLARED oversize body is refused up front with 413
# (test above); a body that never declares a length can only be caught as
# its chunks arrive, inside _BudgetedReceive — a branch the chunked test
# above never reaches, because there the BUDGET runs out first.


def test_chunked_body_over_the_per_request_cap_is_cut_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shrink the cap (160 MB of real chunks is not a unit test) and leave the
    # budget room for BOTH chunks — 600 + 600 fits 1200 — so the cut can only
    # be the per-request cap, never the budget branch tested above.
    monkeypatch.setattr(auth_mod, "MAX_REQUEST_BYTES", 1000)
    app = _BodyApp()
    app.hold.set()   # never hold: one request at a time
    mw = BearerAuthMiddleware(
        app, tokens={TOKEN_A: "agent-a"}, max_inflight_body_bytes=1200,
    )

    async def scenario() -> list[_Message]:
        # No Content-Length, so the up-front 413 check has nothing to look
        # at; the cap has to bite as the second chunk arrives.
        cut = await _drive_body(mw, [600, 600], declare=False)
        # 900 fits the budget only if the cut-off request's 600 charged
        # bytes were released when it finished (600 + 900 > 1200).
        await _drive_body(mw, [900], declare=False)
        return cut

    cut = asyncio.run(scenario())
    assert app.disconnects == 1
    assert app.received == [600, 900]   # first request cut after one chunk
    assert cut[0]["status"] == 200      # the app answered; no 413 mid-body
    assert mw._budget._used == 0        # nothing left charged


def test_chunked_body_under_the_per_request_cap_is_delivered_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Control for the above: same shape, same cap, total under it.
    monkeypatch.setattr(auth_mod, "MAX_REQUEST_BYTES", 1000)
    app = _BodyApp()
    app.hold.set()
    mw = BearerAuthMiddleware(
        app, tokens={TOKEN_A: "agent-a"}, max_inflight_body_bytes=1200,
    )

    asyncio.run(_drive_body(mw, [400, 400], declare=False))
    assert app.disconnects == 0
    assert app.received == [800]
