"""Per-agent identity for MCP requests.

Every bearer token maps to a named agent ("claude-code", "codex", ...).
The auth middleware resolves the presented token to its agent name and
stashes it in a ContextVar so downstream code (tools, audit log, git
commit trailers) can attribute the action without threading an extra
argument through every call.

ContextVars are the right carrier here: the ASGI server runs each
request in its own task with its own context copy, and
``anyio.to_thread.run_sync`` (which app.py uses to offload blocking
tool bodies) copies the caller's context into the worker thread. If a
transport ever dispatches tool execution onto a *different* task that
was not spawned from the request task, the value degrades to the
default "unknown" rather than mis-attributing to another agent — the
fail-safe direction.
"""
from __future__ import annotations

import re
from contextvars import ContextVar

# Default is "unknown", never an agent name: code that runs outside a
# request (startup, tests, background workers) must not impersonate.
AGENT_VAR: ContextVar[str] = ContextVar("brain_mcp_agent", default="unknown")


def current_agent() -> str:
    """Agent name for the request being served, or "unknown" outside one."""
    return AGENT_VAR.get()


# Agent names end up in git commit messages, frontmatter provenance and
# JSONL logs — keep them boring: lowercase slug, no leading separator,
# max 32 chars. Same shape as vault slugs.
_NAME_RE: re.Pattern[str] = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

# Matches the existing BRAIN_MCP_BEARER_TOKEN floor (openssl rand
# -base64 32 yields 44 chars; 24 is the minimum we accept anywhere).
_MIN_TOKEN_CHARS: int = 24


def parse_token_spec(raw: str) -> dict[str, str]:
    """Parse the ``BRAIN_MCP_TOKENS`` env format into token -> agent name.

    Format: ``name=token,name2=token2`` — whitespace around names, tokens
    and commas is tolerated, as is a trailing comma. Raises RuntimeError
    with a clear message on any violation; this runs at config-load time,
    so failing loudly (same style as load_config) beats limping along
    with a half-parsed token set.
    """
    tokens: dict[str, str] = {}
    names_seen: set[str] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue  # tolerate "a=x,,b=y" and trailing commas
        name, sep, token = entry.partition("=")
        name = name.strip()
        token = token.strip()
        if not sep:
            raise RuntimeError(
                f"BRAIN_MCP_TOKENS entry {entry!r} is not name=token "
                "(expected format: name=token,name2=token2)"
            )
        if not _NAME_RE.match(name):
            raise RuntimeError(
                f"BRAIN_MCP_TOKENS agent name {name!r} is invalid "
                "(want: lowercase [a-z0-9_-], starts alphanumeric, max 32 chars)"
            )
        # "=" in the token would be ambiguous with the name separator and
        # "," can't survive the split above; refuse both so a mangled env
        # line fails loudly instead of silently truncating a token.
        if "=" in token or "," in token:
            raise RuntimeError(
                f"BRAIN_MCP_TOKENS token for {name!r} contains '=' or ',' — "
                "tokens must be plain (generate with: openssl rand -hex 32)"
            )
        if len(token) < _MIN_TOKEN_CHARS:
            raise RuntimeError(
                f"BRAIN_MCP_TOKENS token for {name!r} is shorter than "
                f"{_MIN_TOKEN_CHARS} characters (generate with: openssl rand -hex 32 — "
                "base64 output ends in '=', which BRAIN_MCP_TOKENS rejects)"
            )
        if name in names_seen:
            raise RuntimeError(f"BRAIN_MCP_TOKENS has duplicate agent name {name!r}")
        if token in tokens:
            raise RuntimeError(
                f"BRAIN_MCP_TOKENS reuses one token for {tokens[token]!r} and "
                f"{name!r} — identity would be ambiguous"
            )
        names_seen.add(name)
        tokens[token] = name
    if not tokens:
        raise RuntimeError("BRAIN_MCP_TOKENS is set but contains no name=token entries")
    return tokens
