#!/usr/bin/env bash
#
# Sync the public ``template`` branch from the current ``main`` working tree.
#
# Run from the main branch (clean or dirty — the script copies the working
# tree, not the committed state). It creates the ``template`` branch on
# first run (as an orphan, so none of your personal history is included),
# or refreshes it on subsequent runs.
#
# Files synced from main:
#   - scripts/, pyproject.toml, uv.lock
#   - AGENTS.md, CLAUDE.md, mcp/, .devcontainer/, .gitignore, .gitattributes
#   - .env.example, .claude/CODEBASE.md, .claude/{hooks,memory,patterns}/.gitkeep
#   - .obsidian/{app,appearance,core-plugins,graph}.json
#   - knowledge/index/Note Template.md
#
# Files sourced from _template/ on main (overlay specific to the public branch):
#   - README.md, LICENSE, TODO.md, WORK_LOG.md, CONTRIBUTING.md
#   - knowledge/index/Home.md            (from _template/Home.md)
#   - .github/PULL_REQUEST_TEMPLATE.md   (from _template/PULL_REQUEST_TEMPLATE.md)
#
# Files NEVER copied (your private content):
#   - inbox/**, archive/**, logs/**, metadata/**
#   - knowledge/index/<everything except Home.md and Note Template.md>
#   - knowledge/concepts/**
#   - hand-written notes anywhere else under knowledge/
#   - .env, .obsidian/workspace.json
#
# After the script runs, push to your public remote with the line it
# prints at the end.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "error: must be on main, currently on '$CURRENT_BRANCH'"
    exit 1
fi

if [ ! -d "_template" ]; then
    echo "error: _template/ directory missing — the public overlay should live there"
    exit 1
fi

# Use a worktree so the main checkout isn't disturbed.
WORKTREE=$(mktemp -d -t brain-template-sync-XXXXXX)
echo "worktree: $WORKTREE"
trap 'git worktree remove --force "$WORKTREE" 2>/dev/null || true; rm -rf "$WORKTREE"' EXIT

# Create or attach to the template branch.
if git show-ref --verify --quiet refs/heads/template; then
    git worktree add "$WORKTREE" template
    # Clean the worktree so removed files on main translate to removals
    # on template. Keep .git intact.
    find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
else
    # First run: create an orphan branch with no shared history.
    git worktree add --detach "$WORKTREE"
    (
        cd "$WORKTREE"
        git checkout --orphan template
        git rm -rf --cached . 2>/dev/null || true
        find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
    )
fi

# Framework files to copy verbatim from main.
FRAMEWORK_PATHS=(
    "scripts"
    "pyproject.toml"
    "uv.lock"
    "AGENTS.md"
    "CLAUDE.md"
    "mcp"
    ".devcontainer"
    ".gitignore"
    ".gitattributes"
    ".env.example"
    ".claude/CODEBASE.md"
    ".claude/hooks/.gitkeep"
    ".claude/memory/.gitkeep"
    ".claude/patterns/.gitkeep"
    ".obsidian/app.json"
    ".obsidian/appearance.json"
    ".obsidian/core-plugins.json"
    ".obsidian/graph.json"
    "knowledge/index/Note Template.md"
)

copy_path() {
    local src="$1"
    local dest="$WORKTREE/$1"
    if [ ! -e "$src" ]; then
        echo "  skip (missing): $src"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    if [ -d "$src" ]; then
        cp -R "$src" "$(dirname "$dest")/"
    else
        cp "$src" "$dest"
    fi
}

echo "copying framework files..."
for path in "${FRAMEWORK_PATHS[@]}"; do
    copy_path "$path"
done

# Overlay: public-facing files that live under _template/ on main and
# get copied into their target locations on the template branch.
echo "applying _template overlay..."
mkdir -p "$WORKTREE/knowledge/index" "$WORKTREE/.github"
cp "_template/README.md"                "$WORKTREE/README.md"
cp "_template/LICENSE"                  "$WORKTREE/LICENSE"
cp "_template/TODO.md"                  "$WORKTREE/TODO.md"
cp "_template/WORK_LOG.md"              "$WORKTREE/WORK_LOG.md"
cp "_template/CONTRIBUTING.md"          "$WORKTREE/CONTRIBUTING.md"
cp "_template/Home.md"                  "$WORKTREE/knowledge/index/Home.md"
cp "_template/PULL_REQUEST_TEMPLATE.md" "$WORKTREE/.github/PULL_REQUEST_TEMPLATE.md"

# Ensure the empty directories the framework expects exist, with .gitkeep
# files so git tracks them.
echo "seeding .gitkeep markers..."
for dir in \
    "inbox" \
    "archive/raw" "archive/processed" "archive/failed" \
    "knowledge/index" "knowledge/concepts" "knowledge/projects" \
    "knowledge/university" "knowledge/research" "knowledge/people" \
    "knowledge/organisations" "knowledge/notes" \
    "logs" "metadata" \
    ; do
    mkdir -p "$WORKTREE/$dir"
    if [ -z "$(ls -A "$WORKTREE/$dir" 2>/dev/null)" ]; then
        touch "$WORKTREE/$dir/.gitkeep"
    fi
done

# Safety net: refuse to commit if any obviously personal path is present.
echo "running safety scan..."
for forbidden in \
    "$WORKTREE/inbox/university" \
    "$WORKTREE/archive/raw/university" \
    "$WORKTREE/archive/processed/university" \
    "$WORKTREE/knowledge/index/university" \
    "$WORKTREE/metadata/index.jsonl" \
    "$WORKTREE/metadata/embeddings.npy" \
    "$WORKTREE/_template" \
    "$WORKTREE/.env" \
    "$WORKTREE/copilot" \
    "$WORKTREE/.obsidian/plugins" \
    ; do
    if [ -e "$forbidden" ]; then
        echo "ABORT: personal path leaked into template: $forbidden"
        exit 1
    fi
done

cd "$WORKTREE"
git add -A

if git diff --cached --quiet; then
    echo "nothing to sync (template already up-to-date with main)"
    exit 0
fi

git commit -m "sync framework from main ($(date +%Y-%m-%d))"

echo
echo "template branch updated. To publish:"
echo
echo "  # first time: create the public repo (do this once)"
echo "  gh repo create <your-user>/brain-template --public --source=$ROOT --remote=public --push=false"
echo
echo "  # add the remote on the main repo (do this once)"
echo "  git -C $ROOT remote add public git@github.com:<your-user>/brain-template.git"
echo
echo "  # push the template branch as main on the public repo"
echo "  git -C $ROOT push public template:main"
echo
