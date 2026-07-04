"""Runtime configuration for the MCP server.

All knobs come from environment variables so the systemd unit and any
local-test invocations stay declarative. No hardcoded paths or tokens.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .identity import parse_token_spec


# Roots the note tools (create/replace/append) may WRITE under. inbox/ is
# deliberately NOT here: it is reachable only through drop_inbox_file's own
# resolve_inbox() path, which refuses to overwrite — listing it here would
# let replace_note destroy a pending (typically uncommitted) source file.
# knowledge/meetings and knowledge/assistant are the entity-memory areas:
# meeting notes plus the assistant's own inbox/archive/digests.
WRITE_ALLOW_PREFIXES: Final[tuple[str, ...]] = (
    "knowledge/notes",
    "knowledge/projects",
    "knowledge/research",
    "knowledge/people",
    "knowledge/organisations",
    "knowledge/university",
    "knowledge/meetings",
    "knowledge/assistant",
)

# The assistant's standing profile of the user. Written through its own
# size-capped tool (see profile_max_bytes below), not the general note
# verbs, so it can't balloon into a dumping ground.
PROFILE_NOTE_PATH: Final[str] = "knowledge/assistant/PROFILE.md"

# Concept notes have a special write tool that only edits the user
# section below the AUTO-GENERATED-END marker. That tool may write
# under this prefix; no other tool may.
CONCEPT_WRITE_PREFIX: Final[str] = "knowledge/concepts"

# Paths that even READ tools refuse. Mostly system / secret locations.
# archive/raw and archive/processed ARE readable; they're just write-
# protected.
DENY_READ_PREFIXES: Final[tuple[str, ...]] = (
    ".git",
    ".env",
    ".env.example",  # innocuous, but still treat env-shaped files as off-limits
    "logs",          # noisy, regenerable, not interesting to agents
)

# Files an agent should never see, even when allowed-by-prefix.
DENY_READ_NAMES: Final[frozenset[str]] = frozenset({
    ".env",
    ".env.local",
    "id_rsa", "id_ed25519",  # ssh keys
    # The embeddings index holds the full text of every processed chunk.
    # It lives under metadata/ (read-allowed) but the only intended path
    # to that text is vault_search, which applies the read gate per hit.
    # Deny direct reads so the gate stays meaningful.
    "embeddings.npy", "embeddings_meta.jsonl",
})

# Reads are ALLOWLIST-based: only these prefixes (and the root doc files
# below) are readable. Everything else — scripts/, mcp/, .claude/, .ssh/,
# .gitconfig, any *.env, the server's own code — is refused, so a
# compromised agent can't enumerate or read secrets that happen to live
# in the repo. The DENY_* lists above still apply *within* allowed areas.
READ_ALLOW_PREFIXES: Final[tuple[str, ...]] = (
    "knowledge",
    "archive",
    "inbox",
    "metadata",
)

# Individual root-level files an agent may read (vault docs only).
READ_ALLOW_ROOT_FILES: Final[frozenset[str]] = frozenset({
    "README.md", "AGENTS.md", "CLAUDE.md", "TODO.md", "WORK_LOG.md",
})

# Per-tool input size caps. Numbers tuned to user feedback (May 2026).
MAX_INBOX_BYTES: Final[int] = 100 * 1024 * 1024   # 100 MB; covers 31 MB PDFs + headroom
MAX_NOTE_BYTES: Final[int] = 5 * 1024 * 1024      # 5 MB Markdown is ~2.5 M words
# Hard ceiling on a single request body, checked from Content-Length before
# the handler buffers anything. A 100 MB inbox upload is ~133 MB once base64-
# encoded inside the JSON-RPC envelope, so allow headroom above that.
MAX_REQUEST_BYTES: Final[int] = 160 * 1024 * 1024

# Token-bucket parameter for write tools. Read/search tools have their own
# buckets, hardcoded in mcp_server.tools (search 60/min, reads 120/min).
WRITE_RATE_PER_MINUTE: Final[int] = 30   # ~one write every 2 seconds sustained


@dataclass(frozen=True)
class ServerConfig:
    """Everything the server reads from the environment, validated once."""

    vault_root: Path             # absolute path to the brain repo root
    tokens: tuple[tuple[str, str], ...]  # (token, agent_name) pairs; >=1 required
    bind_host: str               # 127.0.0.1 for dev, 0.0.0.0 only behind CF Tunnel
    bind_port: int
    git_push_on_write: bool      # commit + push every write back to origin
    git_remote: str              # name of the remote to push to (default: origin)
    git_branch: str              # which branch to commit on (default: main)
    log_level: str               # uvicorn / app log level
    allowed_hosts: tuple[str, ...]   # extra Host-header values to allow (DNS-rebind guard)
    profile_max_bytes: int       # size cap on PROFILE_NOTE_PATH writes


def load_config() -> ServerConfig:
    """Read env vars, validate, return a frozen config. Raises on missing required values."""
    vault_root_str = os.environ.get("BRAIN_MCP_VAULT_ROOT")
    if not vault_root_str:
        raise RuntimeError("BRAIN_MCP_VAULT_ROOT must be set (absolute path to the brain repo root)")
    vault_root = Path(vault_root_str).expanduser().resolve()
    if not (vault_root / ".git").exists():
        raise RuntimeError(f"BRAIN_MCP_VAULT_ROOT={vault_root} is not a git repository")

    # Two token sources, both optional individually, at least one required:
    # BRAIN_MCP_TOKENS carries named per-agent tokens; the legacy single
    # BRAIN_MCP_BEARER_TOKEN becomes the agent "default". They may coexist
    # (e.g. while migrating clients), but a shared token or a second
    # "default" would make identity ambiguous — refuse those.
    token_pairs: list[tuple[str, str]] = []
    raw_spec = os.environ.get("BRAIN_MCP_TOKENS")
    if raw_spec:
        token_pairs.extend(parse_token_spec(raw_spec).items())

    bearer = os.environ.get("BRAIN_MCP_BEARER_TOKEN")
    if bearer:
        if len(bearer) < 24:
            raise RuntimeError(
                "BRAIN_MCP_BEARER_TOKEN must be at least 24 characters "
                "(generate with: openssl rand -hex 32)"
            )
        if any(token == bearer for token, _ in token_pairs):
            raise RuntimeError(
                "BRAIN_MCP_BEARER_TOKEN duplicates a token in BRAIN_MCP_TOKENS — "
                "identity would be ambiguous"
            )
        if any(agent == "default" for _, agent in token_pairs):
            raise RuntimeError(
                "agent name 'default' in BRAIN_MCP_TOKENS clashes with "
                "BRAIN_MCP_BEARER_TOKEN (which is always agent 'default'); rename it"
            )
        token_pairs.append((bearer, "default"))

    if not token_pairs:
        raise RuntimeError(
            "set BRAIN_MCP_TOKENS (name=token,name2=token2) and/or "
            "BRAIN_MCP_BEARER_TOKEN (generate with: openssl rand -hex 32 — "
            "base64 output ends in '=', which BRAIN_MCP_TOKENS rejects)"
        )

    return ServerConfig(
        vault_root=vault_root,
        tokens=tuple(token_pairs),
        profile_max_bytes=int(os.environ.get("BRAIN_PROFILE_MAX_BYTES", "4096")),
        bind_host=os.environ.get("BRAIN_MCP_BIND_HOST", "127.0.0.1"),
        bind_port=int(os.environ.get("BRAIN_MCP_BIND_PORT", "8765")),
        git_push_on_write=os.environ.get("BRAIN_MCP_GIT_PUSH_ON_WRITE", "1") == "1",
        allowed_hosts=tuple(
            h.strip() for h in os.environ.get("BRAIN_MCP_ALLOWED_HOSTS", "").split(",")
            if h.strip()
        ),
        git_remote=os.environ.get("BRAIN_MCP_GIT_REMOTE", "origin"),
        git_branch=os.environ.get("BRAIN_MCP_GIT_BRANCH", "main"),
        log_level=os.environ.get("BRAIN_MCP_LOG_LEVEL", "info"),
    )
