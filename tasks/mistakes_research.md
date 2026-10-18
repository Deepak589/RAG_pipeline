# Research-backed fixes for mistakes.md (pulled 2026-10-17)

Maps the Stage-6 cost mistakes to recent papers and the concrete change to make.

---

## M1 + M6 — screening on the full set, no spend guard
**Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices** — arXiv:2609.25645
- Each pipeline config = a bandit arm; each question = one noisy observation. Evaluate the
  highest-index arm on a batch, update a conjugate Gaussian posterior, recompute indices, stop
  when a fully-evaluated arm tops the index or the budget is spent.
- Reports near-zero simple regret at **1–2% of exhaustive eval cost**; adaptive stop fires at 1–10%.
- Anytime recommendation = max(posterior mean − 1 SD) → structurally prevents "winner" calls off a
  handful of questions.
- Caveat: offline replays of precomputed response matrices, Gaussian surrogate, price proxies not
  measured tokens. Treat the % as directional.
- Action: replace "30-Q smoke → full set" with batched sequential eval (B=8). Arms = knob configs.
  Set a per-example $ cost from token prices. Log `usage` per call into the eval JSON.

**Are We Testing Too Much? Benchmark Compression survey** — arXiv:2610.18248
**tinyBenchmarks** — arXiv:2402.14992 (IRT-based ~100-item subsets)
- Background for picking the slice itself rather than hand-stratifying 30 Q.

**How Benchmark Prediction from Fewer Data Misses the Mark** — arXiv:2506.07673
- Counter-evidence: subset-prediction methods degrade exactly on the near-ties you care about in a
  knob sweep. Read before trusting a small slice to rank two close configs.

---

## M1 again — one-knob-at-a-time was the wrong design
**Post-Generation Verification Dominates Retrieval Optimization: A 2^4 Factorial Ablation** — arXiv:2609.35774
- 16 configs × 24 queries × 2 models = ~760 runs. Welch t-test, Cohen's d, bootstrap CI (5k).
- Finding: completeness check d=+0.48; ToC-guided retrieval d=+0.22 **at zero extra LLM calls**;
  agentic search d=+0.14 (borderline); section expansion d=−0.01 (null).
- Architectural point: a factorial design gets main effects **and** interactions from the same call
  budget that OAT spends on main effects alone. 4 knobs × 24 Q beats 4 sequential sweeps × 160 Q.
- Caveat: one corpus, single LLM judge, no human agreement check, 2–4 queries per type.
- Action: for the next sweep, define 3–4 binary knobs and run the full 2^k on the slice.

---

## M4 — re-judging identical answers on Sonnet
**On Cost-Effective LLM-as-a-Judge Improvement Techniques** — arXiv:2604.13717 (RewardBench 2, n=1753)
- **Mini + criteria injection at k=8 = 81.5%, matching full-model k=8 at ~1/4 the cost.**
  Best mini result: Haiku 4.5 + criteria, k=8 → 85.8% at 1.3× baseline cost.
- Criteria injection (one category-specific sentence in the judge prompt) is nearly free: +3pp at k=1.
- Ensembling k=1→8 helps small models most (+19.1pp nano, +14.4pp mini, +9.8pp full).
  Two-thirds of the gain is already at k=3.
- Adaptive escalation on mini-variance **underperformed** plain mini k=8 — don't build the router.
- Temperature 0 is not deterministic; ensembling still gained 4.6pp over it.
- Note: ensemble via the API `n` parameter → input tokens charged once, output scales with k.
- Action: judge = Haiku + criteria sentence + k=3 for screening. Cache verdicts by
  (question_id, sha256(answer)). Paper says nothing about caching — that one is just engineering.

**Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge** — arXiv:2605.10805
- Reasoning vs non-reasoning judge routing as distributionally-robust constrained optimization.
  Abstract only; no public numbers. Lower priority given 2604.13717's negative routing result.

---

## M5 — gold set generated twice, no output gate
**How Fine-Grained Should a RAG Benchmark Be? (HieraRAG)** — arXiv:2606.12789
- Three generation dimensions, each at 3 granularities: Question Complexity, Answer Type,
  Linguistic Variation (lexical overlap with the source).
- Human validation on 110 stratified pairs: 98.2% answers acceptable, 90.9% hallucination-free,
  Fleiss κ=0.47. Fine-grained category agreement only 28.6% → fine taxonomies don't survive contact.
- **It has no filtering step and reports no rejection rate.** The validator you need is still yours
  to write. Use medium granularity (4 categories/dimension), not fine.
- Action: the LV dimension is the direct fix for the "according to the passage" failures — generate
  with an explicit low-lexical-overlap constraint and reject high-overlap items automatically.

**RAQAG** (ScienceDirect, S095070512600568X) — automated Q&A dataset generation framework, for the
  generate→validate loop structure.

---

## M8 + M9 — rules after `JSON:`, and JSON-format prompt under structured output
**The Format Tax** — arXiv:2604.03616
**Let Me Speak Freely?** — arXiv:2408.02442 (the original result: format restriction degrades reasoning)
**Empirical Study for Structured Output Control in LLMs** — arXiv:2606.09395
**When Correct Isn't Usable: Structured Output Reliability in Small Models** — arXiv:2605.02363
- Confirms the mechanism behind M9: asking for a text format *and* constraining decoding fights
  itself. Schema-only, no format instructions in the prompt.
- M8's "constraints before the output cue" rule is correct and is standard prompt-position evidence;
  the papers add that you must still hard-validate, since format compliance ≠ content compliance.

---

## M11 — bought an experiment before running the free diagnostic
**RAGXplain: From Explainable Evaluation to Actionable Guidance of RAG Pipelines** — arXiv:2505.13538
- Turns eval metrics into stage-level actions. Closest thing to "which knob should I even buy".
**When Failures Propagate: Causal Failure Attribution in Agentic RAG** — arXiv:2608.20627
- Injects a known fault at a hop, re-runs downstream, checks if the diagnoser finds it.
- Coverage-based diagnosis: 0.91 at hop 1, **0.00 at hops 2 and 3**. Frozen-hop counterfactual probe
  got 0.67 on depth-2 content corruption where coverage got 0.00.
- Lesson for CRAG: post-hoc trace analysis loses the signal once the downstream trajectory changes.
  Log per-hop diagnostics **at execution time**, not reconstructed after.
- Caveat: n=18 and n=3 on the method comparison. Treat as a hypothesis.
**Doctor-RAG: Failure-Aware Repair Framework for Agentic RAG** — arXiv:2604.00865
**Not All RAGs Are Created Equal: Component-Wise Empirical Study** — arXiv:2605.14503

---

## No paper needed
M2, M3, M7, M10, M12, M13 are engineering discipline, not research gaps. The rules already in
mistakes.md are correct. M14 is a gold-set validator rule — fold it into the M5 automatic gate.

---

## Priority order
1. Judge: Haiku + criteria + k=3 + verdict cache (M4) — biggest $ cut, one afternoon.
2. Factorial 2^k on the slice instead of OAT sweeps (M1) — biggest design win, free.
3. Per-call `usage` logging + up-front cost estimate (M6) — prerequisite for 4.
4. Gittins/bandit sequential eval (M1/M6) — only worth it once 1–3 are in place.
5. LV-constrained generation + hard validator gate (M5/M14).
