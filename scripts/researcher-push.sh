#!/usr/bin/env bash
# researcher-push.sh
# Usage: researcher-push.sh <file-path>
# Called automatically by Claude Code PostToolUse hook when the mars-researcher
# agent writes a file under docs/ideas/. Creates a researcher-idea/<slug> branch,
# commits the artifact, and pushes — with SSH-agent fallback.
set -euo pipefail

FILE_PATH="${1:-}"

if [[ -z "$FILE_PATH" ]]; then
  echo "[researcher-push] ERROR: no file path provided" >&2
  exit 1
fi

# Only act on files under docs/ideas/
if [[ "$FILE_PATH" != *"docs/ideas/"* ]]; then
  exit 0
fi

# Derive slug from filename, e.g. docs/ideas/magnetic-shield-restoration.md → magnetic-shield-restoration
SLUG=$(basename "$FILE_PATH" .md)
BRANCH="researcher-idea/${SLUG}"

ROOT=$(git -C "$(dirname "$FILE_PATH")" rev-parse --show-toplevel)
cd "$ROOT"

echo "[researcher-push] Branch: $BRANCH"
echo "[researcher-push] File:   $FILE_PATH"

# Create or switch to the branch (from main so it's a clean base)
if git show-ref --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH" "$(git remote show origin | awk '/HEAD branch/ {print $NF}')" 2>/dev/null \
    || git checkout -b "$BRANCH"
fi

git add "$FILE_PATH"
git commit -m "research: add ${SLUG} direction document

Auto-committed by mars-researcher agent.

Co-Authored-By: Dr. Martian <mars-researcher@claude.agent>"

# --- push with SSH-agent fallback ---
push_with_fallback() {
  if git push -u origin "$BRANCH" 2>/tmp/push_err; then
    echo "[researcher-push] Pushed successfully."
    return 0
  fi

  echo "[researcher-push] Push failed, attempting SSH-agent fallback..." >&2
  cat /tmp/push_err >&2

  # Start ssh-agent and add the project key
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_rsa.sameeerkashyap 2>/dev/null \
    || echo "[researcher-push] WARNING: could not add ~/.ssh/id_rsa.sameeerkashyap" >&2

  # Retry
  if git push -u origin "$BRANCH"; then
    echo "[researcher-push] Pushed successfully after SSH fallback."
    return 0
  fi

  echo "[researcher-push] ERROR: push failed even after SSH fallback." >&2
  return 1
}

push_with_fallback
