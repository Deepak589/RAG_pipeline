#!/usr/bin/env bash
# SessionEnd hook: (1) generate a handoff summary from the transcript via headless
# claude, (2) update the graphify knowledge graph. Blocking (runs before the next
# SessionStart on /clear).
set -uo pipefail

# Self-locate (rename-proof): .claude/ holds this script; project is its parent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SUMMARY_FILE="$SCRIPT_DIR/SESSION_SUMMARY.md"
LOG="$SCRIPT_DIR/session_hooks.log"
export PATH="/opt/anaconda3/bin:/Users/deepakkatukuri/.local/bin:$PATH"

# Recursion guard: the headless `claude` below runs in this project and would
# re-fire SessionEnd. The sentinel is inherited by that subprocess; bail if set.
if [ -n "${CLAUDE_SESSION_SUMMARY_RUNNING:-}" ]; then exit 0; fi
export CLAUDE_SESSION_SUMMARY_RUNNING=1

input=$(cat)
transcript=$(printf '%s' "$input" | jq -r '.transcript_path // empty')
reason=$(printf '%s' "$input" | jq -r '.reason // empty')
echo "=== SessionEnd $(date) reason=$reason transcript=$transcript ===" >> "$LOG"

# 1. Summary from transcript
if [ -n "$transcript" ] && [ -f "$transcript" ]; then
  summary=$(claude -p --model haiku "You are writing a handoff note for the NEXT Claude Code session in this project. The session transcript (JSONL) is piped as input. Produce concise, specific, technical markdown with these sections: ## What was done, ## Key decisions, ## Files changed, ## Current state, ## Next steps. No preamble, no filler." < "$transcript" 2>>"$LOG")
  if [ -n "$summary" ]; then
    { echo "# Session Summary"; echo "_auto-generated $(date) — reason: ${reason}_"; echo; printf '%s\n' "$summary"; } > "$SUMMARY_FILE.tmp" && mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"
    echo "summary written ($(wc -c < "$SUMMARY_FILE" | tr -d ' ') bytes)" >> "$LOG"
  else
    echo "WARN: empty summary (see errors above)" >> "$LOG"
  fi
else
  echo "WARN: no transcript at '$transcript'" >> "$LOG"
fi

# 2. Update graphify knowledge graph
cd "$PROJECT_DIR" && graphify update . >> "$LOG" 2>&1
echo "graphify update exit=$?" >> "$LOG"

exit 0
