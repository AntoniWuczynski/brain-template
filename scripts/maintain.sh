#!/usr/bin/env bash
#
# Deterministic vault maintenance: consolidate assistant memory, then lint
# the vault. Both steps are cron/launchd-safe (deterministic, exit 0), so
# this is the single entry point for a scheduler AND for running by hand.
#
#   scripts/maintain.sh              # real run
#   scripts/maintain.sh --dry-run    # plan only (no writes)
#
# Scheduling examples live next to this repo:
#   - macOS  : mcp_server/launchd/com.brain.maintenance.plist
#   - Linux  : mcp_server/systemd/brain-maintenance.{service,timer}
#
# Caveat: consolidate/sweep take no cross-process lock against a running MCP
# server (only the server's in-process write lock exists), so schedule this
# for a quiet hour when the server is idle (see scripts/README.md).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

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

echo "[maintain $(ts)] consolidate assistant memory"
# $DRY is unquoted so an empty value expands to nothing (no phantom "" arg).
run_py scripts/consolidate.py $DRY || echo "[maintain] consolidate failed (non-fatal)"

echo "[maintain $(ts)] sweep (vault lint)"
# --dry-run has no meaning for sweep; it's read-only unless --write-report.
if [ -z "$DRY" ]; then
    run_py scripts/sweep.py --write-report || echo "[maintain] sweep failed (non-fatal)"
else
    run_py scripts/sweep.py || echo "[maintain] sweep failed (non-fatal)"
fi

echo "[maintain $(ts)] done"
exit 0
