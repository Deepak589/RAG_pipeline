
# Stage 6 — 40pct knob sweep (2026-10-07)

1. [x] A1 retry-on-refusal (48Q slice) → no change, +26% latency. DROP.
2. [x] top_k=10 (48Q slice) → correct_refusal 94.7% (neg-4 hallucinated). DROP, keep k=5.
3. [x] Diagnostics in crag_eval rows: gold_verdicts, gold_in_kept, answer_pos, answer.
4. [x] B2 sparse OR (`--sparse-mode or`). Found plainto_tsquery matched 0/141 positives → sparse side was dead.
       Full 160: e2e 72.3 vs 73.0 control (noise), correct_refusal 94.7 (neg-4). DROP as-is (ts_rank_cd no IDF).
5. [x] A2 set grading (`--grade-mode set`) → e2e 72.3 (noise); answered precision 95.3 vs 89.6, refusals 20 vs 12, 23% faster.
6. [x] Root cause: of 11 control failures where grader dropped gold, 6 have answer past char 1200 (grader sees text[:1200]); 0 visible.
7. [x] A3 `--grader-chars 4000` full 160 → verify: refused_on_hit/answered_wrong down, correct_refusal 100.
8. [x] Gold set expansion qa_expand.py → +100 factual, +40 paraphrase, +20 near-miss negatives (verified unanswerable).
9. [x] Re-run control + best knob on expanded set.
10. [ ] B1 blurbs — DEFERRED (35k LLM calls + 2h re-embed; not today).

## Review (2026-10-07)
- WINNER A3 grader_chars 4000 (now default). v4 318Q: e2e 72.8 -> 79.2 (+18 net: 21 fixed / 3 broke),
  holds on old (100->110/141) and new (103->111/138) halves. refused_on_hit 23->11, answered_wrong 20->13.
  Cost: 7.4 -> 11.3 s/q (grader reads ~3x tokens; partly inflated by a parallel run).
- correct_refusal 94.9 -> 92.3: only extra = neg-4 (grounded near-miss: cites real 2023 EU Battery Reg/DSA,
  not "financial"). Both runs answer 2 new negatives that are LABEL bugs (Howmet yes/no "has it disclosed",
  Robinhood/Bitstamp date likely in corpus). Fix labels before trusting refusal %.
- Gold set v4 = qa_stage6_v4.json (318Q, 40pct-only). Sparse side was dead (plainto 0/141) — open issue.
- Next: fix 3 bad negatives; B2 with real BM25 (IDF); generator kept-but-refused (6 cases); B1 blurbs.
