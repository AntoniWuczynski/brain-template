#!/usr/bin/env bash
# Nightly connector pull: fetch, ingest, commit, push. Unattended.
#
#   scripts/nightly_pull.sh granola              # the server: a pure API client
#   scripts/nightly_pull.sh claude_code          # the Mac: transcripts live there
#   scripts/nightly_pull.sh --dry-run granola    # show what a pull would fetch
#
# Each machine pulls only what it can reach. Granola is an HTTPS API, so it
# runs wherever there is a key. Claude Code transcripts sit in
# ~/.claude/projects on the laptop, and shipping them elsewhere to ingest would
# move unredacted session dumps around for nothing.
#
# Two machines ingesting means two writers of metadata/index.jsonl. That is
# safe because the file is append-only and keyed by source hash, the two
# machines pull disjoint sources, and .gitattributes merges it by union — so
# the order below is: get current FIRST, ingest, commit, push. The window in
# which the other machine can land a commit is the length of one ingest.
#
# Same commit discipline as maintain.sh: only on the configured branch, never
# over someone else's staged work, and only the paths this run dirtied.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || { echo "[nightly-pull] cannot resolve repo root from $0" >&2; exit 1; }
cd "$ROOT" || { echo "[nightly-pull] cannot cd to '$ROOT'" >&2; exit 1; }
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY="--dry-run"; shift; fi
[ "$#" -ge 1 ] || { echo "usage: $0 [--dry-run] <connector>..." >&2; exit 2; }

run_py() {
    if [ -x "$ROOT/.venv/bin/python" ]; then "$ROOT/.venv/bin/python" "$@"; else uv run --no-sync python "$@"; fi
}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Everything an ingest can write that git tracks.
SURFACE=(archive metadata/index.jsonl metadata/connectors knowledge)
dirty_surface() {
    git -C "$ROOT" status --porcelain -uall -z -- "${SURFACE[@]}" 2>/dev/null \
        | tr '\0' '\n' | cut -c4- | grep -v '^$' | LC_ALL=C sort
}

EXPECTED="${BRAIN_MCP_GIT_BRANCH:-main}"
REMOTE="${BRAIN_MCP_GIT_REMOTE:-origin}"
BRANCH=$(git -C "$ROOT" symbolic-ref --short HEAD 2>/dev/null) || BRANCH="(detached HEAD)"
if [ "$BRANCH" != "$EXPECTED" ]; then
    echo "[nightly-pull $(ts)] branch '$BRANCH' is not '$EXPECTED' — not running"; exit 1
fi

if [ -n "$DRY" ]; then
    for c in "$@"; do run_py scripts/pull.py "$c" --dry-run; done
    exit 0
fi

if ! git -C "$ROOT" diff --cached --quiet; then
    echo "[nightly-pull $(ts)] index has staged changes — someone is mid-commit, not running"; exit 1
fi

# Get current before writing anything. --autostash carries a human's
# uncommitted Obsidian edits across the rebase; a conflict aborts the run
# rather than leaving a half-rebased vault for the morning.
echo "[nightly-pull $(ts)] sync with $REMOTE/$EXPECTED"
if ! git -C "$ROOT" pull -q --rebase --autostash "$REMOTE" "$EXPECTED"; then
    git -C "$ROOT" rebase --abort >/dev/null 2>&1
    echo "[nightly-pull $(ts)] could not rebase onto $REMOTE/$EXPECTED — not running; resolve by hand"; exit 1
fi

BEFORE=$(mktemp "${TMPDIR:-/tmp}/brain-nightly.XXXXXX"); AFTER=$(mktemp "${TMPDIR:-/tmp}/brain-nightly.XXXXXX")
trap 'rm -f "$BEFORE" "$AFTER"' EXIT
dirty_surface > "$BEFORE"

FAILED=0
for c in "$@"; do
    echo "[nightly-pull $(ts)] pull + ingest: $c"
    run_py scripts/pull.py "$c" --then-ingest || { echo "[nightly-pull] $c failed (continuing)"; FAILED=1; }
done

dirty_surface > "$AFTER"
NEW=$(LC_ALL=C comm -13 "$BEFORE" "$AFTER")
if [ -z "$NEW" ]; then
    echo "[nightly-pull $(ts)] nothing new — no commit"; exit "$FAILED"
fi
if ! printf '%s\n' "$NEW" | tr '\n' '\0' | xargs -0 git -C "$ROOT" add --; then
    git -C "$ROOT" reset -q >/dev/null 2>&1
    echo "[nightly-pull $(ts)] git add failed — nothing committed"; exit 1
fi
N=$(printf '%s\n' "$NEW" | grep -c .)
git -C "$ROOT" commit -q -m "vault: nightly pull ($*), $N paths" \
    || { echo "[nightly-pull $(ts)] commit refused (the tripwire, or nothing staged) — left staged for a human"; exit 1; }

# Push, and if another writer landed a commit during the ingest, take it and
# try once more.
#
# That is the normal case, not the rare one: an ingest regenerates the shared
# derived notes (hundreds of knowledge/concepts/*), and it ends with an index
# build that takes minutes, so anything else that touched the vault meanwhile
# conflicts with it. The first live run, 2026-09-20, committed 389 paths for
# one meeting and lost exactly this race. The job's commit holds only ingest
# output, which the next run rebuilds, so where it conflicts the remote side
# wins ("-X ours" names the upstream during a rebase) and the run's own new
# files land regardless.
#
# Only when this commit is the SOLE one ahead. A rebase replays every
# unpushed commit, and preferring the remote over a human's would silently
# discard their side — so with anything else in the stack, a conflict stops.
if ! git -C "$ROOT" push -q "$REMOTE" "$EXPECTED" 2>/dev/null; then
    echo "[nightly-pull $(ts)] push rejected — rebasing once"
    git -C "$ROOT" fetch -q "$REMOTE" "$EXPECTED"
    AHEAD=$(git -C "$ROOT" rev-list --count "$REMOTE/$EXPECTED..HEAD" 2>/dev/null || echo 0)
    # ${X[@]+...} below: bash 3.2 (macOS /bin/bash, which launchd runs) treats
    # an empty array as unbound under `set -u`.
    STRATEGY=()
    [ "$AHEAD" = "1" ] && STRATEGY=(-X ours)
    if git -C "$ROOT" pull -q --rebase --autostash ${STRATEGY[@]+"${STRATEGY[@]}"} "$REMOTE" "$EXPECTED"; then
        git -C "$ROOT" push -q "$REMOTE" "$EXPECTED" || { echo "[nightly-pull] second push failed — commit is local"; exit 1; }
    else
        git -C "$ROOT" rebase --abort >/dev/null 2>&1
        echo "[nightly-pull $(ts)] rebase conflict — commit is local; resolve by hand"; exit 1
    fi
fi
echo "[nightly-pull $(ts)] done: $N paths committed and pushed"
exit "$FAILED"
