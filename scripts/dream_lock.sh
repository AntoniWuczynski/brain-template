#!/usr/bin/env bash
#
# One dream pass at a time, scheduled or manual.
#
#   scripts/dream_lock.sh acquire <owner>   # 0 = held, 1 = busy (prints the holder)
#   scripts/dream_lock.sh release <owner>   # 0 = released, 1 = held by someone else
#
# <owner> is `scheduled` (scripts/dream.sh) or `manual` (the skill's step 0).
# The lock lives where the scheduled pass runs. When the vault is served from
# another machine, set BRAIN_DREAM_LOCK_HOST (in the environment or this repo's
# .env) to that machine's ssh name, and both commands run there over ssh. Unset,
# the lock is local. A lock older than BRAIN_DREAM_LOCK_STALE seconds (default
# 7200, past dream.sh's 5400s runner cap) belongs to a run that died, and the
# next acquire breaks it.
set -uo pipefail

CMD="${1:-}"
OWNER="${2:-}"
case "$CMD:$OWNER" in
    acquire:scheduled | acquire:manual | release:scheduled | release:manual) ;;
    *) echo "usage: $0 acquire|release scheduled|manual" >&2; exit 2 ;;
esac

# `-` not `:-`: an explicitly empty value (the remote call below) means local,
# without falling back to .env.
if [ -z "${BRAIN_DREAM_LOCK_HOST+set}" ]; then
    ROOT=$(cd "$(dirname "$0")/.." && pwd)
    BRAIN_DREAM_LOCK_HOST=$(sed -n 's/^BRAIN_DREAM_LOCK_HOST=//p' "$ROOT/.env" 2>/dev/null | tail -n 1 | tr -d "\"'")
fi
if [ -n "$BRAIN_DREAM_LOCK_HOST" ]; then
    # Ship this script over stdin, so the host needs no deployed copy of it.
    exec ssh "$BRAIN_DREAM_LOCK_HOST" "BRAIN_DREAM_LOCK_HOST= bash -s -- $CMD $OWNER" <"$0"
fi

LOCK="${BRAIN_DREAM_LOCK_DIR:-$HOME/.cache/brain-dream/run.lock}"
STALE="${BRAIN_DREAM_LOCK_STALE:-7200}"
NOW=$(date +%s)

# "<owner> <epoch>". A lock whose owner file is not written yet (the instant
# between mkdir and the write below) is dated by the directory's mtime, so a
# concurrent acquire never mistakes it for an ancient one.
holder() {
    cat "$LOCK/owner" 2>/dev/null \
        || echo "unknown $(stat -c %Y "$LOCK" 2>/dev/null || stat -f %m "$LOCK" 2>/dev/null || echo "$NOW")"
}

case "$CMD" in
    acquire)
        mkdir -p "$(dirname "$LOCK")"
        if ! mkdir "$LOCK" 2>/dev/null; then
            read -r who since <<<"$(holder)"
            if [ $((NOW - ${since:-0})) -gt "$STALE" ]; then
                echo "[dream-lock] breaking a stale lock held by $who since $(date -u -r "${since:-0}" +%FT%TZ 2>/dev/null || date -u -d "@${since:-0}" +%FT%TZ)"
                # Rename first: only one concurrent breaker's mv succeeds, so a
                # loser can never delete the lock the winner just took.
                mv "$LOCK" "$LOCK.stale.$$" 2>/dev/null && rm -rf "$LOCK.stale.$$"
                mkdir "$LOCK" 2>/dev/null || { echo "[dream-lock] lost the race for the lock"; exit 1; }
            else
                echo "[dream-lock] a dream pass is already running: $who since $(date -u -r "${since:-0}" +%FT%TZ 2>/dev/null || date -u -d "@${since:-0}" +%FT%TZ)"
                exit 1
            fi
        fi
        echo "$OWNER $NOW" >"$LOCK/owner"
        echo "[dream-lock] held by $OWNER"
        ;;
    release)
        [ -d "$LOCK" ] || { echo "[dream-lock] not held"; exit 0; }
        read -r who _ <<<"$(holder)"
        if [ "$who" != "$OWNER" ]; then
            echo "[dream-lock] held by $who, not $OWNER — left in place"
            exit 1
        fi
        rm -rf "$LOCK"
        echo "[dream-lock] released by $OWNER"
        ;;
esac
