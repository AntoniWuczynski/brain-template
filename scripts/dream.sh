#!/usr/bin/env bash
#
# Dream pass scheduler entry point: run the deterministic gate; if enough
# new information landed, launch a headless LLM session that executes
# .claude/skills/dream-pass/SKILL.md (subscription auth, no API credits).
#
#   scripts/dream.sh                          # gate, then dream if warranted
#   BRAIN_DREAM_RUNNER=noop scripts/dream.sh  # gate only (testing)
#   BRAIN_DREAM_RUNNER=codex scripts/dream.sh # alternate runner
#
# Scheduling: mcp_server/launchd/com.brain.dream.plist (05:00 daily —
# deliberately an hour after com.brain.maintenance so consolidate/sweep
# finish first). Every run appends to logs/dream-<UTC>.log (AGENTS.md
# rule 5). Always exits 0: cron/launchd must not treat a skipped or
# failed dream as a scheduler failure — the log and the dream-stalled
# sweep check are the signal.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# launchd runs with a minimal PATH; make sure claude/codex/uv are findable.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/dream-$TS.log"
mkdir -p "$ROOT/logs"

run_py() {
    if [ -x "$ROOT/.venv/bin/python" ]; then
        "$ROOT/.venv/bin/python" "$@"
    else
        uv run --no-sync python "$@"
    fi
}

RUNNER="${BRAIN_DREAM_RUNNER:-claude}"
MAX_TURNS="${BRAIN_DREAM_MAX_TURNS:-50}"

echo "[dream $TS] gate check" >>"$LOG"
if run_py scripts/dream_gate.py >>"$LOG" 2>&1; then
    echo "[dream] gate passed — launching runner: $RUNNER" >>"$LOG"
    case "$RUNNER" in
        claude)
            claude -p "/dream-pass" \
                --allowedTools "mcp__brain__*" "Read" "Grep" "Glob" \
                    "Bash(uv run --no-sync python scripts/dream_gate.py*)" \
                    "Bash(.venv/bin/python scripts/dream_gate.py*)" \
                --max-turns "$MAX_TURNS" >>"$LOG" 2>&1 \
                || echo "[dream] claude run failed (state not advanced; will retry tomorrow)" >>"$LOG"
            ;;
        codex)
            codex exec "Read .claude/skills/dream-pass/SKILL.md in this repo and execute the dream pass exactly as written." \
                >>"$LOG" 2>&1 \
                || echo "[dream] codex run failed (state not advanced; will retry tomorrow)" >>"$LOG"
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
