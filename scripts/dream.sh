#!/usr/bin/env bash
#
# Dream pass scheduler entry point: run the deterministic gate; if enough
# new information landed, launch a headless LLM session that executes
# .claude/skills/dream-pass/SKILL.md (subscription auth, no API credits).
#
#   scripts/dream.sh                          # gate, then dream if warranted
#   BRAIN_DREAM_RUNNER=noop scripts/dream.sh  # gate only (testing)
#   BRAIN_DREAM_RUNNER=codex scripts/dream.sh # alternate runner
#   BRAIN_DREAM_TIMEOUT=1800 scripts/dream.sh # wall-clock cap, seconds
#
# Scheduling: mcp_server/launchd/com.brain.dream.plist (05:00 daily —
# deliberately an hour after com.brain.maintenance so consolidate/sweep
# finish first). Every run appends to logs/dream-<UTC>.log (AGENTS.md
# rule 5). Always exits 0: cron/launchd must not treat a skipped or
# failed dream as a scheduler failure — the log and the dream-stalled
# sweep check are the signal.
set -uo pipefail

# Guard both steps: without `set -e`, a failed subshell would leave ROOT empty
# and `cd ""` returns 0 without moving, so the dream would run against the
# wrong tree and log to $ROOT/logs = /logs.
ROOT=$(cd "$(dirname "$0")/.." && pwd) || { echo "[dream] cannot resolve repo root from $0" >&2; exit 1; }
cd "$ROOT" || { echo "[dream] cannot cd to '$ROOT'" >&2; exit 1; }

# launchd runs with a minimal PATH; make sure claude/codex/uv are findable.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/dream-$TS.log"
mkdir -p "$ROOT/logs"

# Step 7 of the skill adjudicates the contradiction findings and hands the
# survivors to `dream_gate.py --propose <file>`, so the session needs one
# writable path. Deliberately OUTSIDE the vault: the file is scratch, not a
# note, and the runner must not be able to write into knowledge/ except
# through the MCP tools. Under $HOME rather than /tmp — a predictable path
# in a world-writable directory is somebody else's to plant. Created here
# so the session never needs a shell to make it.
SCRATCH="$HOME/.cache/brain-dream"
mkdir -p "$SCRATCH"

run_py() {
    if [ -x "$ROOT/.venv/bin/python" ]; then
        "$ROOT/.venv/bin/python" "$@"
    else
        uv run --no-sync python "$@"
    fi
}

RUNNER="${BRAIN_DREAM_RUNNER:-claude}"
MAX_TURNS="${BRAIN_DREAM_MAX_TURNS:-50}"
# --max-turns bounds turns, not wall time: a stalled MCP call or a model
# outage would otherwise leave the session running for days, and launchd
# will not start tomorrow's copy while one is alive. Cap it. SIGTERM
# first, SIGKILL 60s later, so the timeout itself cannot wedge either.
TIMEOUT_SECS="${BRAIN_DREAM_TIMEOUT:-5400}"
TIMEOUT_BIN=""
for candidate in timeout gtimeout; do
    if command -v "$candidate" >/dev/null 2>&1; then
        TIMEOUT_BIN="$candidate"
        break
    fi
done

# Run the argv under the wall-clock cap when a timeout binary exists;
# without one (stock macOS, no coreutils) run uncapped and say so, rather
# than skipping the dream.
run_capped() {
    if [ -n "$TIMEOUT_BIN" ]; then
        "$TIMEOUT_BIN" --kill-after=60s "$TIMEOUT_SECS" "$@"
    else
        "$@"
    fi
}

report_runner_exit() {
    local name="$1" rc="$2"
    if [ "$rc" -eq 0 ]; then
        return
    fi
    # GNU timeout: 124 = TERM'd at the deadline, 137 = KILL'd after it.
    if [ -n "$TIMEOUT_BIN" ] && { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; }; then
        echo "[dream] $name run killed at the ${TIMEOUT_SECS}s timeout (state not advanced; will retry tomorrow)" >>"$LOG"
    else
        echo "[dream] $name run failed (exit $rc; state not advanced; will retry tomorrow)" >>"$LOG"
    fi
}

echo "[dream $TS] gate check" >>"$LOG"
if run_py scripts/dream_gate.py >>"$LOG" 2>&1; then
    echo "[dream] gate passed — launching runner: $RUNNER (timeout: ${TIMEOUT_BIN:-none} ${TIMEOUT_SECS}s)" >>"$LOG"
    case "$RUNNER" in
        claude)
            # Exactly the tools .claude/skills/dream-pass/SKILL.md uses, named
            # one by one. The brain-wildcard this replaces also handed the
            # session the relation-upsert, profile-update, fact-append and
            # inbox-drop tools, none of which this pass may touch: it
            # PROPOSES graph changes, it never applies them. Replacing whole
            # notes IS needed — step 5 replaces a digest, step 8 merges an
            # entity note, step 9 rewrites questions.md — but it is not how
            # compiled truth is written; that tool splices between the
            # markers and cannot eat a hand-written section.
            # Both spellings of the scratch path — home-relative and
            # absolute ("//" + the expanded $HOME) — because a Write rule
            # that fails to match is a step 7 that silently cannot run, and
            # a silent step 7 is exactly the defect this list is fixing.
            run_capped claude -p "/dream-pass" \
                --allowedTools \
                    "mcp__brain__vault_read" \
                    "mcp__brain__vault_search" \
                    "mcp__brain__vault_related" \
                    "mcp__brain__vault_create_note" \
                    "mcp__brain__vault_replace_note" \
                    "mcp__brain__vault_update_compiled_truth" \
                    "Read" "Grep" "Glob" \
                    "Write(~/.cache/brain-dream/**)" \
                    "Write(/$SCRATCH/**)" \
                    "Bash(uv run --no-sync python scripts/dream_gate.py*)" \
                    "Bash(.venv/bin/python scripts/dream_gate.py*)" \
                --max-turns "$MAX_TURNS" >>"$LOG" 2>&1
            report_runner_exit claude $?
            ;;
        codex)
            run_capped codex exec "Read .claude/skills/dream-pass/SKILL.md in this repo and execute the dream pass exactly as written." \
                >>"$LOG" 2>&1
            report_runner_exit codex $?
            ;;
        noop)
            echo "[dream] noop runner: gate result recorded, no session launched" >>"$LOG"
            ;;
        *)
            echo "[dream] unknown BRAIN_DREAM_RUNNER=$RUNNER — nothing launched" >>"$LOG"
            ;;
    esac
else
    RC=$?
    if [ "$RC" -eq 1 ]; then
        echo "[dream] gate: skip (not enough new information)" >>"$LOG"
    else
        echo "[dream] gate error (exit $RC) — see above" >>"$LOG"
    fi
fi
echo "[dream] done" >>"$LOG"
exit 0
