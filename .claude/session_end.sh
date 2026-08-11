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
# The raw JSONL transcript can exceed the model input limit (~200k tokens), and
# headless `claude` also loads the project system prompt + tools (~130k tokens of
# overhead here). So: extract just role+text, keep the MOST RECENT tail, and cap
# hard at 180k chars (~45k tokens) to stay well under the limit.
MAX_CHARS=180000
if [ -n "$transcript" ] && [ -f "$transcript" ]; then
  convo=$(jq -r '
    select(.message)
    | .message as $m
    | (($m.role // "?") + ": " +
       (if ($m.content | type) == "string" then $m.content
        else ([$m.content[]? | (.text // "")] | join(" "))
        end))
  ' "$transcript" 2>>"$LOG" | tail -c "$MAX_CHARS")

  if [ -z "$convo" ]; then
    echo "WARN: empty conversation extracted from transcript" >> "$LOG"
  else
    summary=$(printf '%s' "$convo" | claude -p --model haiku "You are writing a handoff note for the NEXT Claude Code session in this project. A trimmed session transcript (role: text lines, most recent kept) is piped as input. Produce concise, specific, technical markdown with these sections: ## What was done, ## Key decisions, ## Files changed, ## Current state, ## Next steps. No preamble, no filler." 2>>"$LOG")
    # Guard: reject empty output and known overflow/error strings so we never
    # overwrite a good summary with garbage.
    if [ -z "$summary" ]; then
      echo "WARN: empty summary (see errors above); keeping previous summary" >> "$LOG"
    elif printf '%s' "$summary" | head -1 | grep -qiE 'prompt is too long|request is too large|rate limit|error'; then
      echo "WARN: summary looks like an error ('$(printf '%s' "$summary" | head -c 80)'); keeping previous summary" >> "$LOG"
    else
      { echo "# Session Summary"; echo "_auto-generated $(date) — reason: ${reason}_"; echo; printf '%s\n' "$summary"; } > "$SUMMARY_FILE.tmp" && mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"
      echo "summary written ($(wc -c < "$SUMMARY_FILE" | tr -d ' ') bytes)" >> "$LOG"
    fi
  fi
else
  echo "WARN: no transcript at '$transcript'" >> "$LOG"
fi

# 2. Update graphify knowledge graph
cd "$PROJECT_DIR" && graphify update . >> "$LOG" 2>&1
echo "graphify update exit=$?" >> "$LOG"

exit 0
