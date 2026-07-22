#!/usr/bin/env bash
# SessionStart hook: inject previous session summary so Claude reads it first.
set -uo pipefail

# Guard: skip when invoked from inside the session-end summary generation
# (headless `claude` in this project would otherwise re-trigger hooks).
[ -n "${CLAUDE_SESSION_SUMMARY_RUNNING:-}" ] && exit 0

# Self-locate (rename-proof): summary lives next to this script in .claude/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_FILE="$SCRIPT_DIR/SESSION_SUMMARY.md"
[ -f "$SUMMARY_FILE" ] || exit 0

jq -n --rawfile s "$SUMMARY_FILE" \
  '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:("=== PREVIOUS SESSION SUMMARY (read first) ===\n\n" + $s)}}'
