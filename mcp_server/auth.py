"""Bearer-token auth middleware.

Layered in front of the MCP mount in app.py. Cloudflare Access is the
outer ring of auth (SSO / service tokens); this is the inner ring that
ensures even a request that slipped past CF can't reach the tools
without the right token.

We use constant-time comparison and don't put the expected token into
any log line or error response.
"""
from __future__ import annotations

import hmac
import logging
import threading
from collections import OrderedDict

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import MAX_REQUEST_BYTES
from .identity import AGENT_VAR

log = logging.getLogger(__name__)


# Endpoints that don't require auth. /health is the only one — useful
# for systemd's start-up probe and Cloudflare's origin health check.
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health"})

# The Streamable-HTTP transport routes by this header, so a valid token
# alone doesn't say which agent a request belongs to.
_SESSION_HEADER: bytes = b"mcp-session-id"

# Ceiling on remembered session -> agent bindings. One live client holds
# one session, so this is orders of magnitude above the real fleet; the
# cap only exists so a client that leaks sessions can't grow the map
# without bound. Oldest binding is evicted first.
_MAX_TRACKED_SESSIONS: int = 1024

# Ceiling on request-body bytes held in memory across ALL in-flight requests.
# MAX_REQUEST_BYTES bounds one body; the per-tool concurrency guards only
# engage inside the tool, i.e. once a request has been fully received and
# JSON-parsed, so without a second ceiling N authenticated clients can pin
# N * MAX_REQUEST_BYTES before any guard or rate limit can refuse them. Two
# max-size bodies at once is far above real use (the 160 MB request is a
# base64 inbox upload, and one agent uploads one file at a time), and the
# cap itself can't come down: MAX_INBOX_BYTES of 100 MB is ~133 MB encoded.
_MAX_INFLIGHT_BODY_BYTES: int = 2 * MAX_REQUEST_BYTES


class _BodyBudget:
    """Thread-safe counter of request-body bytes currently held in memory.

    Lives on the middleware instance (one per process) and is touched from
    every request task, so a plain lock guards it — every operation is an
    integer compare, never awaits.
    """

    def __init__(self, limit: int = _MAX_INFLIGHT_BODY_BYTES) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._used = 0

    def has_room(self, size: int) -> bool:
        """Would ``size`` more bytes fit right now? Advisory: two requests
        can both see room and only one of them get it — ``reserve`` is the
        authority, this only avoids reading a body that clearly can't fit."""
        with self._lock:
            return self._used + size <= self._limit

    def reserve(self, size: int) -> bool:
        with self._lock:
            if self._used + size > self._limit:
                return False
            self._used += size
            return True

    def release(self, size: int) -> None:
        with self._lock:
            self._used = max(0, self._used - size)


class _SessionOwners:
    """Bounded, thread-safe ``mcp-session-id`` -> agent-name map.

    The ASGI app can be driven from more than one event loop / thread
    (uvicorn workers, tests), so the map is guarded by a plain lock: every
    operation is a dict access, never awaits.
    """

    def __init__(self, limit: int = _MAX_TRACKED_SESSIONS) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._owners: OrderedDict[str, str] = OrderedDict()

    def claim(self, session_id: str, agent: str) -> str:
        """Bind ``session_id`` to ``agent`` if it is unbound; return the
        owning agent either way (so the caller can compare)."""
        with self._lock:
            owner = self._owners.get(session_id)
            if owner is None:
                self._owners[session_id] = agent
                if len(self._owners) > self._limit:
                    self._owners.popitem(last=False)
                return agent
            self._owners.move_to_end(session_id)
            return owner

    def release(self, session_id: str) -> None:
        with self._lock:
            self._owners.pop(session_id, None)


class BearerAuthMiddleware:
    """Pure-ASGI bearer-token gate.

    Implemented as raw ASGI rather than Starlette's BaseHTTPMiddleware:
    BaseHTTPMiddleware buffers the response body and is known to break
    SSE / streaming, which the MCP Streamable-HTTP transport relies on.
    On the happy path this passes scope/receive/send through untouched.

    Cloudflare Access is the outer auth ring (SSO / service tokens);
    this is the inner ring that gates anything reaching the origin.

    Each token maps to a named agent. On success the agent name is set
    in ``identity.AGENT_VAR`` *in this request's task* before the app
    runs, so tool bodies and the audit log can attribute the call — the
    contextvar survives ``anyio.to_thread.run_sync`` offloads because
    anyio copies the caller's context into the worker thread. Known
    caveat: if a transport ever dispatches tool execution onto a task
    not derived from the request task, identity degrades to "unknown"
    rather than mis-attributing to another agent — the fail-safe
    direction.

    Sessions are bound to their opener. The Streamable-HTTP transport
    routes by ``mcp-session-id``, so without this any holder of any valid
    token could attach to a session another agent opened. A session id is
    claimed the first time it is seen — in the ``initialize`` response
    that mints it, so the binding is made at creation — and a request
    presenting another agent's session id is refused with 403.
    """

    def __init__(
        self,
        app: ASGIApp,
        tokens: dict[str, str] | None = None,
        max_inflight_body_bytes: int = _MAX_INFLIGHT_BODY_BYTES,
    ) -> None:
        """``tokens`` maps token -> agent name. At least one is required.
        ``max_inflight_body_bytes`` bounds the request-body bytes this
        process buffers at once (see ``_MAX_INFLIGHT_BODY_BYTES``)."""
        self.app = app
        if not tokens:
            raise ValueError("BearerAuthMiddleware needs a non-empty tokens= map")
        # Pre-encode once; compare as bytes so a non-ASCII header can't
        # raise TypeError inside compare_digest.
        self._tokens: tuple[tuple[bytes, str], ...] = tuple(
            (token.encode("utf-8"), agent) for token, agent in tokens.items()
        )
        self._sessions = _SessionOwners()
        self._budget = _BodyBudget(max_inflight_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if scope.get("path") in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        header = ""
        for k, v in scope.get("headers") or []:
            if k == b"authorization":
                header = v.decode("latin-1")
                break

        presented = header[len("Bearer "):].strip() if header[:7].lower() == "bearer " else ""
        presented_b = presented.encode("utf-8", "replace")
        # Each comparison is constant-time; iterating ALL tokens (no early
        # break, token count is tiny) keeps the loop's timing independent
        # of which token matched. `presented == ""` for a missing or
        # malformed header always fails every digest check.
        agent: str | None = None
        for expected_b, name in self._tokens:
            if hmac.compare_digest(presented_b, expected_b):
                agent = name
        if not presented or agent is None:
            log.warning("auth: rejected request from %s", _client_ip(scope))
            await _send_unauthorized(scope, send)
            return

        # A session belongs to the agent that opened it. Claiming here also
        # covers a session first seen on a request (nothing else can have
        # claimed it), so an id is never usable by two agents.
        session_id = _header_value(scope, _SESSION_HEADER)
        if session_id:
            owner = self._sessions.claim(session_id, agent)
            if owner != agent:
                log.warning(
                    "auth: agent %s presented a session owned by %s (from %s)",
                    agent, owner, _client_ip(scope),
                )
                await _send_forbidden(scope, send)
                return

        # Reject oversize bodies up front (before the handler buffers/decodes).
        declared = _declared_length(scope)
        if declared is not None and declared > MAX_REQUEST_BYTES:
            await _send_too_large(scope, send)
            return
        # ...and refuse one that would not fit alongside the bodies already
        # in flight, rather than buffering it and finding out inside the tool.
        if declared is not None and not self._budget.has_room(declared):
            log.warning("auth: body of %d bytes refused, budget full", declared)
            await _send_overloaded(scope, send)
            return

        # Content-Length can be absent (chunked transfer), and a present one
        # is only a claim. Wrap receive to enforce both ceilings on the bytes
        # actually delivered; on exceed, hand the app an http.disconnect so
        # it aborts instead of buffering the rest.
        capped: _BudgetedReceive | None = (
            _BudgetedReceive(receive, MAX_REQUEST_BYTES, self._budget)
            if scope["type"] == "http" else None
        )
        recv: Receive = receive if capped is None else capped

        # A new session's id first appears in the response that mints it,
        # so watch the response headers when the request carried none.
        # Header-inspecting only: every message is forwarded untouched and
        # nothing is buffered, so SSE streaming is unaffected.
        snd = send if session_id else self._claiming_send(send, agent)

        # Identity must be set BEFORE the app runs and in THIS task, so
        # the value propagates into thread offloads (see class docstring).
        # Reset afterwards out of tidiness — each request runs in its own
        # context copy anyway, so leakage across requests can't happen.
        var_token = AGENT_VAR.set(agent)
        try:
            await self.app(scope, recv, snd)
        finally:
            AGENT_VAR.reset(var_token)
            if capped is not None:
                capped.release()
            # DELETE is the transport's session-termination verb, and only
            # the owner gets this far — drop the binding so the map tracks
            # live sessions.
            if session_id and scope.get("method") == "DELETE":
                self._sessions.release(session_id)

    def _claiming_send(self, send: Send, agent: str) -> Send:
        """Wrap an ASGI send so a session id minted in the response is
        bound to ``agent`` as it goes out."""

        async def wrapped(message: Message) -> None:
            if message.get("type") == "http.response.start":
                for k, v in message.get("headers") or []:
                    key: bytes = k
                    if key.lower() == _SESSION_HEADER:
                        value: bytes = v
                        self._sessions.claim(value.decode("latin-1").strip(), agent)
                        break
            await send(message)

        return wrapped


class _BudgetedReceive:
    """ASGI receive wrapper enforcing two ceilings on one request's body:
    it may not exceed ``cap`` bytes, and every byte delivered is charged to
    ``budget`` so concurrent bodies can't pin unbounded memory. Over either
    ceiling the app is handed an ``http.disconnect`` so it aborts instead of
    buffering the rest. ``release()`` returns this request's charge."""

    def __init__(self, receive: Receive, cap: int, budget: _BodyBudget) -> None:
        self._receive = receive
        self._cap = cap
        self._budget = budget
        self._total = 0
        self._charged = 0

    async def __call__(self) -> Message:
        msg = await self._receive()
        if msg.get("type") == "http.request":
            size = len(msg.get("body", b""))
            self._total += size
            if self._total > self._cap or not self._budget.reserve(size):
                return {"type": "http.disconnect"}
            self._charged += size
        return msg

    def release(self) -> None:
        self._budget.release(self._charged)
        self._charged = 0


def _declared_length(scope: Scope) -> int | None:
    """Content-Length as an int, or None when absent or unparseable. An
    unparseable value is not treated as oversize: _BudgetedReceive is the
    authority on the bytes that actually arrive."""
    for k, v in scope.get("headers") or []:
        if k == b"content-length":
            try:
                return int(v)
            except ValueError:
                return None
    return None


def _header_value(scope: Scope, name: bytes) -> str:
    """First value of header ``name`` (ASGI lowercases header names), or ""."""
    for k, v in scope.get("headers") or []:
        if k == name:
            value: bytes = v
            return value.decode("latin-1").strip()
    return ""


def _client_ip(scope: Scope) -> str:
    """Real client IP for logging. Behind cloudflared the socket peer is
    always 127.0.0.1; the true client is in CF-Connecting-IP. The header is
    attacker-controllable if the origin is ever bound to a routable address
    without the tunnel, so only trust it when it parses as an IP address —
    otherwise fall back to the real socket peer rather than logging an
    attacker-chosen string."""
    import ipaddress

    for k, v in scope.get("headers") or []:
        if k == b"cf-connecting-ip":
            if not isinstance(v, bytes):
                break  # malformed ASGI header value: fall through to the socket peer
            candidate = v.decode("latin-1").strip()
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                break  # malformed header value: fall through to the socket peer
            return candidate
    client = scope.get("client")
    return client[0] if client else "?"


async def _send_unauthorized(scope: Scope, send: Send) -> None:
    await _send_status(scope, send, 401, b'{"error":"unauthorized"}', ws_code=1008)


async def _send_forbidden(scope: Scope, send: Send) -> None:
    await _send_status(scope, send, 403, b'{"error":"forbidden"}', ws_code=1008)


async def _send_too_large(scope: Scope, send: Send) -> None:
    await _send_status(scope, send, 413, b'{"error":"request too large"}', ws_code=1009)


async def _send_overloaded(scope: Scope, send: Send) -> None:
    # Not the request's fault (it is under the per-request cap) — the server
    # is already holding as much body as it will hold. Retryable.
    await _send_status(scope, send, 503, b'{"error":"server busy"}', ws_code=1013)


async def _send_status(scope: Scope, send: Send, status: int, body: bytes, ws_code: int) -> None:
    # Bare status only — nothing about the expected token or issuer.
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": ws_code})
        return
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})
