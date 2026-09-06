#!/usr/bin/env bash
#
# Deterministic vault maintenance: propose wikilink-derived relations,
# consolidate assistant memory, rebuild the search index, lint the vault,
# rotate telemetry, then commit whatever the run wrote. Every step is
# cron/launchd-safe (deterministic, exit 0, no interactive git), so this is
# the single entry point for a scheduler AND for running by hand.
#
#   scripts/maintain.sh              # real run
#   scripts/maintain.sh --dry-run    # plan only (no writes, no commit)
#
# Scheduling examples live next to this repo:
#   - macOS  : mcp_server/launchd/com.brain.maintenance.plist
#   - Linux  : mcp_server/systemd/brain-maintenance.{service,timer}
#
# Caveat: consolidate/sweep take no cross-process lock against a running MCP
# server (only the server's in-process write lock exists), so schedule this
# for a quiet hour when the server is idle (see scripts/README.md).
set -uo pipefail

# Guard both steps: without `set -e`, a failed subshell would leave ROOT
# empty, and `cd ""` is a no-op that returns 0 — so the run would carry on
# in whatever directory the scheduler happened to hand us, with
# "$ROOT/.venv/bin/python" resolving to /.venv/bin/python and every `git -C
# ""` failing, all of it hidden behind the non-fatal echoes below.
ROOT=$(cd "$(dirname "$0")/.." && pwd) || { echo "[maintain] cannot resolve repo root from $0" >&2; exit 1; }
cd "$ROOT" || { echo "[maintain] cannot cd to '$ROOT'" >&2; exit 1; }

# launchd/systemd run with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin), so
# the `uv` fallback in run_py below is unfindable on a machine with no repo
# venv. Same line as scripts/dream.sh.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Plain string, not an array: macOS ships bash 3.2, where expanding an empty
# array as "${arr[@]}" under `set -u` aborts with "unbound variable" — which
# would kill the scheduled (no-arg) run. A scalar avoids that entirely.
DRY=""
case "${1:-}" in
    "")          ;;                       # normal run
    --dry-run)   DRY="--dry-run" ;;
    *)           echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

# Prefer the repo venv interpreter (what a scheduler has); fall back to uv.
run_py() {
    if [ -x "$ROOT/.venv/bin/python" ]; then
        "$ROOT/.venv/bin/python" "$@"
    else
        uv run --no-sync python "$@"
    fi
}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Working-tree changes under knowledge/, one path per line, sorted for
# comm(1). -uall lists untracked FILES rather than just their directory,
# which is how a fresh archive month or monthly digest shows up. Errors
# (no git, not a repo) yield an empty list, so the commit step no-ops.
dirty_knowledge() {
    git -C "$ROOT" status --porcelain -uall -z -- knowledge 2>/dev/null \
        | tr '\0' '\n' | cut -c4- | grep -v '^$' | LC_ALL=C sort
}

BEFORE_DIRTY=$(mktemp "${TMPDIR:-/tmp}/brain-maintain.XXXXXX")
AFTER_DIRTY=$(mktemp "${TMPDIR:-/tmp}/brain-maintain.XXXXXX")
trap 'rm -f "$BEFORE_DIRTY" "$AFTER_DIRTY"' EXIT
dirty_knowledge > "$BEFORE_DIRTY"

echo "[maintain $(ts)] propose wikilink-derived relations"
# Zero-LLM, deterministic (scripts/ingest_lib/wikilinks.py): drops
# related_to candidates into knowledge/assistant/inbox/ for consolidate
# (below) to promote once a human approves. Order is only about the
# inbox being complete before consolidate reads it — a candidate proposed
# tonight is unapproved with today's created:, so tonight's consolidate
# can only ever WAIT on it; it is a later run, after a human approves it
# or after stale_days, that acts.
run_py -m ingest_lib.wikilinks $DRY || echo "[maintain] wikilink pass failed (non-fatal)"

echo "[maintain $(ts)] propose duplicate-entity merge candidates"
# Zero-LLM, deterministic (scripts/ingest_lib/duplicates.py): reuses sweep's
# entity-duplicate detection to drop merge candidates into
# knowledge/assistant/inbox/ for consolidate (below) to surface for a human
# to approve — the merge itself still has to be performed per AGENTS.md.
run_py -m ingest_lib.duplicates $DRY || echo "[maintain] duplicate-entity pass failed (non-fatal)"

echo "[maintain $(ts)] consolidate assistant memory"
# $DRY is unquoted so an empty value expands to nothing (no phantom "" arg).
run_py scripts/consolidate.py $DRY || echo "[maintain] consolidate failed (non-fatal)"

echo "[maintain $(ts)] rebuild search index (full)"
# The only scheduled FULL rebuild. Nothing else does one: ingest reindexes
# only when it wrote new content, and the MCP refresher's dirty set is
# in-memory, so a note hand-edited, renamed or deleted in Obsidian — or
# written seconds before a server crash — otherwise stays stale in the index
# indefinitely. Rebuilds metadata/embeddings*, which is gitignored, so the
# commit step below is unaffected. ingest.py rejects --dry-run here, so the
# plan-only run just says what it would do.
if [ -z "$DRY" ]; then
    run_py scripts/ingest.py --rebuild-search-index \
        || echo "[maintain] rebuild-search-index failed (non-fatal)"
else
    echo "[maintain] would rebuild the search index (skipped: --dry-run)"
fi

echo "[maintain $(ts)] sweep (vault lint)"
# --dry-run has no meaning for sweep; it's read-only unless --write-report.
if [ -z "$DRY" ]; then
    run_py scripts/sweep.py --write-report || echo "[maintain] sweep failed (non-fatal)"
else
    run_py scripts/sweep.py || echo "[maintain] sweep failed (non-fatal)"
fi

echo "[maintain $(ts)] rotate MCP telemetry logs"
run_py scripts/rotate_logs.py $DRY || echo "[maintain] rotate-logs failed (non-fatal)"

# Commit what this run wrote. Left uncommitted, consolidation's entity
# edits, inbox->archive moves and digests sit on disk until a human notices
# — and on a machine where the MCP server also commits, the next server
# write stages them into an unrelated "mcp(<agent>): ..." commit,
# misattributing them. Same branch discipline as the server
# (git_ops.commit_paths(expected_branch=...) / BRAIN_MCP_GIT_BRANCH): commit
# only on the configured branch, never push (the server's push worker owns
# that), never prompt, and never fail the run.
#
# Only paths this run dirtied are committed: knowledge/ is snapshotted
# before and after, and anything already modified beforehand (a human's
# pending edit) is left alone rather than swept in. logs/,
# metadata/embeddings*, metadata/connections.jsonl and the sweep report are
# gitignored, so knowledge/ is the whole committable surface.
commit_output() {
    command -v git >/dev/null 2>&1 || { echo "[maintain] no git — nothing committed"; return 0; }
    git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 \
        || { echo "[maintain] not a git repo — nothing committed"; return 0; }

    expected="${BRAIN_MCP_GIT_BRANCH:-main}"
    branch=$(git -C "$ROOT" symbolic-ref --short HEAD 2>/dev/null) || branch="(detached HEAD)"
    if [ "$branch" != "$expected" ]; then
        echo "[maintain] branch '$branch' is not '$expected' — nothing committed"
        return 0
    fi
    # A non-empty index is someone else's commit in progress; committing now
    # would fold our output into it (and theirs into ours).
    if ! git -C "$ROOT" diff --cached --quiet; then
        echo "[maintain] index has staged changes — nothing committed"
        return 0
    fi

    dirty_knowledge > "$AFTER_DIRTY"
    new_paths=$(LC_ALL=C comm -13 "$BEFORE_DIRTY" "$AFTER_DIRTY")
    if [ -z "$new_paths" ]; then
        echo "[maintain] no new changes under knowledge/ — nothing to commit"
        return 0
    fi
    if ! printf '%s\n' "$new_paths" | tr '\n' '\0' | xargs -0 git -C "$ROOT" add --; then
        echo "[maintain] git add failed — nothing committed"
        git -C "$ROOT" reset -q >/dev/null 2>&1
        return 0
    fi
    # Commits exactly what was staged above: the index was verified empty first.
    git -C "$ROOT" commit -q -m "maintenance: consolidate + sweep output" \
        || echo "[maintain] git commit failed (non-fatal)"
}

if [ -z "$DRY" ]; then
    echo "[maintain $(ts)] commit maintenance output"
    commit_output
else
    echo "[maintain $(ts)] commit skipped (--dry-run)"
fi

echo "[maintain $(ts)] done"
exit 0
