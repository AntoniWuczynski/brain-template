#!/usr/bin/env bash
# Keep this clone level with the remote. Unattended, every 15 minutes.
#
#   scripts/vault_sync.sh              # a laptop: commit hand edits, rebase, push
#   scripts/vault_sync.sh --pull-only  # the MCP server: fast-forward only
#
# The MCP server commits and pushes its own writes, so there this only
# fast-forwards to pick up what other machines pushed. It never commits (an
# MCP write in flight would be swept in) and never rebases (the server's push
# worker owns its unpushed commits and rebases them itself).
#
# On a laptop it commits whatever changed on the ingest surface — Obsidian
# edits, a hand-run ingest — except knowledge/assistant/, which agents own
# and write through the MCP server. Then it rebases onto the remote and
# pushes. Offline is not a failure: the commit waits for the next run.
#
# A failure raises a macOS notification where osascript exists. A sync that
# fails silently is how this Mac's writes sat unpushed from 2026-09-20 to
# 2026-09-22.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || { echo "[vault-sync] cannot resolve repo root from $0" >&2; exit 1; }
cd "$ROOT" || { echo "[vault-sync] cannot cd to '$ROOT'" >&2; exit 1; }
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() {
    echo "[vault-sync $(ts)] $1"
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"$1\" with title \"brain sync\"" >/dev/null 2>&1
    fi
    exit 1
}

PULL_ONLY=""
[ "${1:-}" = "--pull-only" ] && PULL_ONLY=1

EXPECTED="${BRAIN_MCP_GIT_BRANCH:-main}"
REMOTE="${BRAIN_MCP_GIT_REMOTE:-origin}"
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null) || BRANCH="(detached HEAD)"
[ "$BRANCH" = "$EXPECTED" ] || fail "branch '$BRANCH' is not '$EXPECTED' — not syncing"
GIT_DIR=$(git rev-parse --git-dir)
if [ -d "$GIT_DIR/rebase-merge" ] || [ -d "$GIT_DIR/rebase-apply" ] || [ -f "$GIT_DIR/MERGE_HEAD" ]; then
    fail "a rebase or merge is in progress — resolve it by hand"
fi
# A non-empty index is someone else's commit in progress.
if ! git diff --cached --quiet; then
    echo "[vault-sync $(ts)] index has staged changes — someone is mid-commit, skipping"; exit 0
fi

if [ -n "$PULL_ONLY" ]; then
    git fetch -q "$REMOTE" "$EXPECTED" || fail "fetch from $REMOTE failed"
    if [ "$(git rev-list --count "$REMOTE/$EXPECTED..HEAD")" != "0" ]; then
        echo "[vault-sync $(ts)] local commits not yet pushed — leaving them to the push worker"; exit 0
    fi
    git merge -q --ff-only "$REMOTE/$EXPECTED" || fail "cannot fast-forward to $REMOTE/$EXPECTED"
    echo "[vault-sync $(ts)] up to date at $(git rev-parse --short HEAD)"
    exit 0
fi

# Same surface nightly_pull.sh commits, minus the agent-owned area.
git add -A -- archive metadata/index.jsonl metadata/connectors knowledge ':(exclude)knowledge/assistant' \
    || { git reset -q; fail "git add failed"; }
if ! git diff --cached --quiet; then
    git commit -q -m "sync($(hostname -s)): local edits" || { git reset -q; fail "git commit failed"; }
fi

if ! git fetch -q "$REMOTE" "$EXPECTED" 2>/dev/null; then
    echo "[vault-sync $(ts)] $REMOTE unreachable — will retry next run"; exit 0
fi
if ! git rebase -q --autostash "$REMOTE/$EXPECTED" >/dev/null 2>&1; then
    git rebase --abort >/dev/null 2>&1
    fail "rebase onto $REMOTE/$EXPECTED conflicted — local commits kept, resolve by hand"
fi
if [ "$(git rev-list --count "$REMOTE/$EXPECTED..HEAD")" != "0" ]; then
    git push -q "$REMOTE" "$EXPECTED" || fail "push to $REMOTE failed — commits are local"
fi
echo "[vault-sync $(ts)] up to date at $(git rev-parse --short HEAD)"
