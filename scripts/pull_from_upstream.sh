#!/usr/bin/env bash
#
# Pull framework updates from the public template repo (``upstream``
# remote) into the current ``main`` branch.
#
# Mental model: brain-template is the canonical framework. This private
# repo is a downstream consumer. When upstream gets framework fixes or
# new features (your own pushes, or contributors' merged PRs), run this
# to absorb them.
#
# On the very first run after switching to the upstream-as-canonical
# model, pass ``--first-run`` so git lets unrelated histories merge.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "error: must be on main, currently on '$CURRENT_BRANCH'"
    exit 1
fi

if ! git remote get-url upstream >/dev/null 2>&1; then
    echo "error: 'upstream' remote not configured. Add it with:"
    echo "  git remote add upstream git@github.com:<your-user>/brain-template.git"
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree has uncommitted changes. Commit or stash first."
    git status --short
    exit 1
fi

EXTRA_FLAGS=""
if [ "${1:-}" = "--first-run" ]; then
    EXTRA_FLAGS="--allow-unrelated-histories"
    echo "first-run mode: allowing unrelated histories"
fi

git fetch upstream main
echo
echo "diff to be merged (upstream/main vs HEAD):"
git log --oneline HEAD..upstream/main | head -20 || true
echo

read -r -p "Proceed with merge? [y/N] " ans
case "$ans" in
    [yY]*) ;;
    *) echo "aborted."; exit 0 ;;
esac

# shellcheck disable=SC2086
git merge upstream/main $EXTRA_FLAGS

echo
echo "merged. Push to your private origin when ready:"
echo "  git push origin main"
