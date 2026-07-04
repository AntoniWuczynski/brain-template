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

from .config import MAX_REQUEST_BYTES
from .identity import AGENT_VAR

log = logging.getLogger(__name__)


# Endpoints that don't require auth. /health is the only one — useful
# for systemd's start-up probe and Cloudflare's origin health check.
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health"})


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
    """

    def __init__(
        self,
        app,
        tokens: dict[str, str] | None = None,
    ) -> None:
        """``tokens`` maps token -> agent name. At least one is required."""
        self.app = app
        if not tokens:
            raise ValueError("BearerAuthMiddleware needs a non-empty tokens= map")
        # Pre-encode once; compare as bytes so a non-ASCII header can't
        # raise TypeError inside compare_digest.
        self._tokens: tuple[tuple[bytes, str], ...] = tuple(
            (token.encode("utf-8"), agent) for token, agent in tokens.items()
        )

    async def __call__(self, scope, receive, send) -> None:
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

        # Reject oversize bodies up front (before the handler buffers/decodes).
        for k, v in scope.get("headers") or []:
            if k == b"content-length":
                try:
                    over = int(v) > MAX_REQUEST_BYTES
                except ValueError:
                    over = False
                if over:
                    await _send_too_large(scope, send)
                    return
                break

        # Content-Length can be absent (chunked transfer). Wrap receive to
        # cap cumulative body bytes; on exceed, hand the app an
        # http.disconnect so it aborts instead of buffering the rest.
        recv = _cap_receive(receive, MAX_REQUEST_BYTES) if scope["type"] == "http" else receive

        # Identity must be set BEFORE the app runs and in THIS task, so
        # the value propagates into thread offloads (see class docstring).
        # Reset afterwards out of tidiness — each request runs in its own
        # context copy anyway, so leakage across requests can't happen.
        var_token = AGENT_VAR.set(agent)
        try:
            await self.app(scope, recv, send)
        finally:
            AGENT_VAR.reset(var_token)


def _cap_receive(receive, cap: int):
    """Wrap an ASGI receive so the cumulative request body can't exceed
    ``cap`` bytes (defends against chunked uploads with no Content-Length)."""
    total = 0

    async def wrapped():
        nonlocal total
        msg = await receive()
        if msg.get("type") == "http.request":
            total += len(msg.get("body", b""))
            if total > cap:
                return {"type": "http.disconnect"}
        return msg

    return wrapped


def _client_ip(scope) -> str:
    """Real client IP for logging. Behind cloudflared the socket peer is
    always 127.0.0.1; the true client is in CF-Connecting-IP. The header is
    attacker-controllable if the origin is ever bound to a routable address
    without the tunnel, so only trust it when it parses as an IP address —
    otherwise fall back to the real socket peer rather than logging an
    attacker-chosen string."""
    import ipaddress

    for k, v in scope.get("headers") or []:
        if k == b"cf-connecting-ip":
            candidate = v.decode("latin-1").strip()
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                break  # malformed header value: fall through to the socket peer
            return candidate
    client = scope.get("client")
    return client[0] if client else "?"


async def _send_unauthorized(scope, send) -> None:
    await _send_status(scope, send, 401, b'{"error":"unauthorized"}', ws_code=1008)


async def _send_too_large(scope, send) -> None:
    await _send_status(scope, send, 413, b'{"error":"request too large"}', ws_code=1009)


async def _send_status(scope, send, status: int, body: bytes, ws_code: int) -> None:
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
