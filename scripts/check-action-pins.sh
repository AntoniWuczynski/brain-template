#!/usr/bin/env bash
# Every `uses:` in a workflow must be pinned to a full 40-hex commit SHA,
# never a mutable version tag — a repointed upstream tag would otherwise
# ship straight into our jobs (the tj-actions/changed-files class of
# supply-chain attack). Dependabot bumps the pins via reviewable PRs.
# A trailing `# vN` comment is fine; a bare `@vN` / `@main` is not.
set -euo pipefail

cd "$(dirname "$0")/.."

# Select every `uses:` line, then drop the ones correctly pinned to a
# 40-hex commit SHA. Anything left is a mutable ref — fail. `|| true`
# keeps the pipeline from tripping `set -e` when the second grep finds
# nothing (the good case).
unpinned="$(grep -rnE 'uses:[[:space:]]' .github/workflows/ \
  | grep -vE '@[0-9a-f]{40}([[:space:]]|$)' || true)"

if [ -n "$unpinned" ]; then
  echo "$unpinned" >&2
  echo "::error::Unpinned GitHub Action above — pin to a full commit SHA." >&2
  exit 1
fi

echo "All workflow actions are SHA-pinned."
