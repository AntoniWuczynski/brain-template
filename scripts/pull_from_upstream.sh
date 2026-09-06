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

# Overlay files: the public branch's copies of these are GENERATED from
# _template/ by push_to_upstream.sh, so upstream is not their author — this
# repo is. Merging them back would overwrite the private vault's own README,
# TODO, WORK_LOG and Home.md with the template's, and would add the
# public-only LICENSE/CONTRIBUTING/PR-template to a private repo that
# deliberately has none. Contributors' fixes to any of these belong in
# _template/ and are applied there by hand.
OVERLAY_PATHS=(
    "README.md"
    "TODO.md"
    "WORK_LOG.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "knowledge/index/Home.md"
    ".github/PULL_REQUEST_TEMPLATE.md"
    ".claude/skills"
)

# --no-commit so the overlay paths can be reset before the merge is recorded;
# --no-ff so there is always a merge commit to attach them to. A conflict here
# is expected (the overlay files differ on both sides), so the failure is
# captured rather than aborting the script.
set +e
# shellcheck disable=SC2086
git merge --no-commit --no-ff $EXTRA_FLAGS upstream/main
MERGE_RC=$?
set -e

if [ ! -e "$(git rev-parse --git-dir)/MERGE_HEAD" ]; then
    # No merge in progress: either already up to date, or git refused to
    # start (e.g. a merge would clobber a local change). Nothing to fix up.
    exit "$MERGE_RC"
fi

echo
echo "keeping this repo's copies of the template overlay files..."
for overlay in "${OVERLAY_PATHS[@]}"; do
    # Drop whatever the merge produced (including a conflicted entry), then
    # restore ours if we have one. Removing first is what makes an
    # upstream-only addition under a synced directory disappear too.
    git rm -rfq --ignore-unmatch -- "$overlay" 2>/dev/null || true
    if [ -n "$(git ls-tree -r --name-only HEAD -- "$overlay")" ]; then
        git checkout HEAD -- "$overlay"
        echo "  kept ours: $overlay"
    else
        echo "  dropped (template-only): $overlay"
    fi
done

UNMERGED=$(git diff --name-only --diff-filter=U)
if [ -n "$UNMERGED" ]; then
    echo
    echo "merge stopped with conflicts outside the overlay set — resolve these,"
    echo "then finish with 'git commit':"
    echo "$UNMERGED"
    exit 1
fi

git commit --no-edit

echo
echo "merged. Push to your private origin when ready:"
echo "  git push origin main"
