# Mistakes — Stage 6 knob sweep session (2026-10-05 → 2026-10-07)

Bill for the session exceeded $20 (~6,500 API calls, ~13–15M input tokens, estimate —
no per-call usage was logged). Most mistakes below are cost mistakes.

## Cost / process

### M1 — Screened every knob on the full set instead of the smoke slice
- **What:** Ran B2, A2, A3 (and control) each on the full 160 Q, then control + A3 on 318 Q.
  The plan explicitly said: 30-Q stratified smoke slice for iteration, full set to confirm.
- **Cost:** ~60% of eval spend was avoidable.
- **Rule:** Screen every new knob on the 30–48 Q slice. Only knobs that pass go to the full set.

### M2 — No timeouts / retries before launching long paid runs
- **What:** One 160-Q run crashed on a single network timeout (default `max_retries=2`).
  The v4 control run hung **3 hours** on a dead HTTPS socket (no read timeout), then had to be killed and rerun.
- **Cost:** ~2 wasted runs (~1,000 calls) + ~3h wall time.
- **Fix applied:** `make_chat_anthropic` now sets `max_retries=6`, `default_request_timeout=120`.
- **Rule:** Before any run > 5 min: bounded timeout + retries, and a watchdog that alerts on no progress (not just on exit).

### M3 — Monitored for "done", not for "stuck"
- **What:** The monitor only watched for `written`/`Traceback`. A hung process produced neither,
  so the hang went unnoticed for hours. stdout was block-buffered, so there were no progress lines to watch either.
- **Rule:** Watch for progress (CPU time / log growth / DB activity). Run Python unbuffered (`-u`) for long jobs.

### M4 — Re-judged identical answers on Sonnet
- **What:** Each A/B run re-invoked the Sonnet judge for answers identical to a previous run.
- **Cost:** ~40–60% of judge calls redundant.
- **Rule:** Cache judge verdicts by (question id, answer hash). Use Haiku as the judge for screening runs.

### M5 — Generated the gold set twice
- **What:** The first full `qa_expand.py` run shipped broken output: paraphrases that were the prompt echoed back,
  or raw JSON; 29/160 questions saying "according to the passage" / accession IDs. The smoke test used 3–4
  questions, too few to catch it, and I didn't read the smoke output critically before scaling.
- **Cost:** One full generation run (~700 calls) thrown away.
- **Rule:** Smoke-test generators on ≥20 items, and gate the output with an automatic check (regex / assert) before scaling up.

### M6 — Ran paid experiments without a spend guard
- **What:** No budget estimate up front, no spend limit, no per-run cost logged.
- **Rule:** State the estimated call count/cost before launching a batch. Log `usage` tokens per run into the eval JSON.

## Correctness / engineering

### M7 — zsh word-splitting broke the first parallel launch
- **What:** `SUB="--conn ... --qa ..."` then `$SUB`: zsh does not word-split, so all 3 runs died on argparse.
- **Rule:** Use arrays in shell: `SUB=(...)` + `"${SUB[@]}"`.

### M8 — Prompt rules appended after `JSON:` were ignored
- **What:** Added FACT_RULES after qa_gen's prompt, which ends with `JSON:`, and Haiku ignored them.
- **Rule:** Put constraints before the content/output cue. Always back prompt rules with a hard validator.

### M9 — Reused a JSON-format prompt with structured output
- **What:** qa_gen's paraphrase prompt asks for `{"question": ...}` JSON. Combined with `with_structured_output`,
  the model returned JSON-in-a-string or the prompt itself.
- **Rule:** With structured output, the prompt must not ask for a text format.

### M10 — Ran 3–4 heavy processes in parallel on the laptop
- **What:** Multiple bge-m3 loads + API threads → a silent process death (memory pressure) and connect timeouts (`Errno 60`).
  Parallel runs also contaminated `sec_per_q` comparisons.
- **Rule:** At most 2 concurrent eval processes. Report latency only from runs that ran alone.

### M11 — Launched top_k=10 before running the cheap diagnostic
- **What:** Ran top_k=10 even though retrieval_hit was already 100% on that slice, so it couldn't help.
  The free check (grader verdict on gold + answer position vs the 1200-char cut) found the real cause
  later and should have come first.
- **Rule:** Diagnose with logged data before buying an experiment. Log per-row diagnostics in every eval by default.

### M12 — Answered a "done" check from stale logs
- **What:** Early in the session, read `a1_*.log` without checking timestamps. They were from a failed earlier run.
- **Rule:** Check file mtimes vs now before trusting a log or result file.

### M13 — Miscounted the gold set ("2 duplicate ids")
- **What:** Assumed 320 questions; the real total was 318, so there were no duplicates. Spent a check on a non-issue.
- **Rule:** Compute counts from the file; don't add up targets in my head.

### M15 — Watchdog never tested, stayed silent through a 50-min stall (2026-10-17)
- **What:** Stall monitor used `declare -A` (bash 4+); it most likely ran under macOS bash 3.2 and errored silently. Expired after 30 min with zero events while the A/B sat stuck.
- **Rule:** Prove a watchdog fires (touch a fake log, wait for the STALL line) before trusting its silence. Plain variables, no assoc arrays.

### M16 — "Bounded" timeout wasn't bounded in practice (2026-10-17)
- **What:** `timeout=120, max_retries=6` bounds ONE call at ~15 min; a question makes up to ~6 calls -> ~90 min worst case. Run crawled through retries on api.anthropic.com for ~50 min.
- **Rule:** Bound wall-clock per question, not per request. Since the LLM cache makes reruns free, kill + rerun beats waiting.

### M14 — Near-miss negative checker missed yes/no questions
- **What:** Accepted "Has Howmet disclosed a material cybersecurity breach?" as unanswerable, but "no, none disclosed" is a valid answer.
  Also verified negatives only against the top-5 parents of the 40pct corpus.
- **Rule:** Reject yes/no and "has X disclosed" style negatives. Re-verify negatives against the full corpus before using them there.
