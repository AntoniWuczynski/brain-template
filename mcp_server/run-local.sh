#!/usr/bin/env bash
#
# Run the brain MCP server locally for use by Claude Code (and other MCP
# clients) on this machine. Binds 127.0.0.1 only.
#
# The bearer token is generated once and persisted to ~/.brain-mcp-token
# so it stays stable across restarts — the token baked into your Claude
# Code config must keep matching. Override the location with
# BRAIN_MCP_TOKEN_FILE.
#
# Never publish that token with `launchctl setenv` (or the shell-profile
# equivalent): a setenv value is inherited by every process in the login
# session, so the 0600 file and 0600 plist stop meaning anything and any app,
# hook or agent on the machine can read a credential that commits — and, with
# BRAIN_MCP_GIT_PUSH_ON_WRITE=1, pushes — to the vault. Clients read the file
# at use time instead, the way the `claude mcp add` line printed below does.
# A second client (Codex, a scheduled job) gets its OWN token: give the server
# BRAIN_MCP_TOKENS="claude=<tok1>,codex=<tok2>" so each agent is separately
# named in the audit log and separately revocable.
#
# Git push is OFF by default for local runs: write tools still commit, but
# nothing is pushed. Set BRAIN_MCP_GIT_PUSH_ON_WRITE=1 to push. Commits only
# land if the vault's checked-out branch matches BRAIN_MCP_GIT_BRANCH
# (default "main", not set here) — on any other branch, writes still land on
# disk but the tool returns committed:false. Export BRAIN_MCP_GIT_BRANCH
# before running this script if you're working from a feature branch.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKEN_FILE="${BRAIN_MCP_TOKEN_FILE:-$HOME/.brain-mcp-token}"

# -s (not -f): regenerate when the file is missing OR empty. An empty file
# can be left behind if `openssl` fails after the `>` redirect already
# created it; without this the server would then boot with an empty token
# and fail config validation forever with no hint to delete the file.
if [ ! -s "$TOKEN_FILE" ]; then
    ( umask 077; openssl rand -base64 32 > "$TOKEN_FILE" )
    echo "generated a new bearer token at $TOKEN_FILE"
fi

export BRAIN_MCP_VAULT_ROOT="$ROOT"
export BRAIN_MCP_BEARER_TOKEN="$(cat "$TOKEN_FILE")"
export BRAIN_MCP_BIND_HOST="${BRAIN_MCP_BIND_HOST:-127.0.0.1}"
export BRAIN_MCP_BIND_PORT="${BRAIN_MCP_BIND_PORT:-8765}"
export BRAIN_MCP_GIT_PUSH_ON_WRITE="${BRAIN_MCP_GIT_PUSH_ON_WRITE:-0}"
export BRAIN_MCP_LOG_LEVEL="${BRAIN_MCP_LOG_LEVEL:-info}"

echo "vault : $ROOT  (branch: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
echo "url   : http://${BRAIN_MCP_BIND_HOST}:${BRAIN_MCP_BIND_PORT}/mcp"
# Print the file, not the secret: echoing the raw token leaks it into
# terminal scrollback and any log a supervisor captures stdout to. The
# registration command below reads it lazily via $(cat ...) at use time.
echo "token : stored in $TOKEN_FILE (0600)"
echo "push  : $BRAIN_MCP_GIT_PUSH_ON_WRITE"
echo
echo "Connect Claude Code (user scope, all sessions):"
echo "  claude mcp add --transport http --scope user \\"
echo "    --header \"Authorization: Bearer \$(cat $TOKEN_FILE)\" \\"
echo "    brain http://${BRAIN_MCP_BIND_HOST}:${BRAIN_MCP_BIND_PORT}/mcp"
echo

cd "$ROOT"
exec uv run python -m mcp_server
