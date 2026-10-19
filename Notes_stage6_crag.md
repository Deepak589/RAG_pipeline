# RAG_dev — Stage 6 notes: CRAG + agentic RAG (production track)

Split from Stage 5. Stage 5 ended at a **trustworthy retrieval baseline: R@5 0.830**
(BGE-M3, pgvector, 160-Q golden set, hybrid @ 70/30). Stage 6 is the jump to a
serving-shaped **agentic** system.

---

## Decisions locked (2026-08-19)

- **Embedder: stay on BGE-M3.** No swap. CRAG is a reasoning layer *on top of*
  retrieval — swapping the embedder is orthogonal and would cost a full re-embed
  of 497k children + a re-baseline for an unmeasured gain. Embedder becomes a
  *later, isolated* A/B (BGE-M3 vs Voyage vs OpenAI) with the CRAG loop held fixed.
  Note: **Anthropic has no embedding model** — it recommends Voyage AI. Claude is
  used for generation + grading only.
- **Generation + grading: Claude API** (`langchain-anthropic` / `ChatAnthropic`).
- **Orchestration: LangGraph** — native state machine for the CRAG loop, reuses
  the Stage-5 `EnsembleRetriever`. Least new surface area.
- **Eval target: BOTH** — refusal (the 19 negatives, open since Stage 3) AND
  answer quality (Claude-as-judge on positives).

## The rule change — "one knob" is not dropped, it is PROMOTED

The instinct "no time for small changes" is real, but the fix is not to ship blind.
The one-knob rule was about the **measured retrieval knobs** (weights, chunking,
embedder). Those still isolate — R@5 is the only judge. What changes:

- **Batch freely, upstream of the metric:** infra, agentic wiring, serving,
  parser fixes. These were never one-knob — they are plumbing measured by "does it
  run / refuse correctly", not by R@5.
- **The eval gate is now automated, not manual.** `crag_eval.py` is the CI gate.
  Ship a batch → the gate runs refusal + answer-quality → regressions surface
  automatically. Fast comes from *automating the check*, not skipping it.

## Why CRAG first (not embedder, not routing)

- CRAG grades retrieval and **refuses when it's bad** → finally exercises the 19
  negatives (the refusal metric open since Stage 3).
- It targets the two known weak spots: paraphrase recall (0.692) gets a second
  chance via query refinement + re-retrieve; wrong retrieval gets caught instead
  of hallucinated over.
- Self-query / metadata routing (Stage-5 shortlist #1) layers on *after* CRAG is
  measured — it is the next knob, not this one.

---

## Architecture — CRAG as a LangGraph state machine

Corrective RAG (Yan et al.) shape, adapted: no web-search fallback (offline eval),
so a hard-fail grade → **refuse**, which is exactly what makes the negatives
measurable.

```
        ┌──────────┐
  query │ retrieve │  hybrid BM25+dense (reuse lc_pipeline.build_hybrid)
        └────┬─────┘  top-K parents, original parent text
             ▼
        ┌──────────┐
        │  grade   │  Claude scores EACH doc: correct | ambiguous | incorrect
        └────┬─────┘  (structured output, enforced JSON)
             ▼
      ┌──────route──────┐
      │                 │
 any correct       none correct
      │                 │
      ▼            ┌─────┴──────┐
 ┌──────────┐   retries<max?  else
 │ generate │      │            │
 └────┬─────┘      ▼            ▼
      │       ┌─────────┐  ┌────────┐
      ▼       │ refine  │  │ refuse │
    (END)     │ query   │  └───┬────┘
              └────┬────┘      ▼
                   │         (END)
                   └──► retrieve (loop, bounded)
```

**State** (`CRAGState`): `query`, `refined_query`, `retrieved` (parent_id, text,
score), `grades`, `kept` (docs graded correct), `answer`, `refused`, `steps`.

**Nodes**
1. `retrieve` — hybrid top-K parents. First pass uses `query`; loop passes use
   `refined_query`.
2. `grade` — Claude, structured output per doc → `correct | ambiguous | incorrect`.
   Aggregate: keep all `correct`. Decision =
   - any correct → `generate`
   - none correct, `steps < MAX_STEPS` → `refine` (rewrite query, re-retrieve)
   - none correct, budget spent → `refuse`
3. `refine_query` — Claude rewrites the query (decompose / disambiguate / expand)
   to fix a retrieval miss. Bounded by `MAX_STEPS` (default 2) — the loop guard
   that keeps "agentic" from becoming "infinite".
4. `generate` — Claude answers from `kept` parent text only; instructed to refuse
   in-band if the kept context still doesn't answer (defense in depth).
5. `refuse` — deterministic "not enough information in the corpus" answer.

**Discipline baked in:** grading and refinement are the ONLY new LLM calls; the
single-shot hybrid stays the fast path (MAX_STEPS=0 collapses CRAG back to
Stage-5 behaviour → the exact A/B control for "did CRAG help?").

## What gets measured (`crag_eval.py`)

Run against `qa_stage5_v3.json` (160 Q), compared to the 0.830 retrieval baseline.

**Refusal (the point of CRAG):**
- *Correct-refusal rate* on 19 negatives — % where CRAG refused. (Baseline
  single-shot: ~0%, it always answers.)
- *False-refusal rate* on positives — % where CRAG wrongly refused an answerable
  question. The cost side; must stay low or CRAG is over-cautious.

**Answer quality (Claude-as-judge):**
- On positives, judge scores the generated answer vs gold (`_answer` where present,
  else gold parent text): `correct | partial | wrong`. Report answer-correctness
  rate + the retrieval-vs-answer gap (retrieval can hit but generation still fail).

**Cost/latency (surfaced, not gated yet):** LLM calls per query, wall time. CRAG
trades latency for correctness — track the trade explicitly.

## Risks / cons (eyes open)

- **Grader is now a silent-failure surface** — a bad grader refuses good docs
  (false refusal) or passes bad ones. Grader prompt is itself a knob; the false-
  refusal metric is its guardrail.
- **Nondeterminism** — every CRAG run can differ. `temperature=0` everywhere;
  eval reports are seeded/logged.
- **Cost** — grading K docs/query = K+ extra tokens. Batch grade in one call.
- **MAX_STEPS is a real knob** — too low = no correction, too high = latency/cost
  blowup. Default 2, tuned against the false-refusal + answer-quality curve.
- **No web fallback offline** — real CRAG re-retrieves from the web on hard-fail;
  here that path is `refuse`. Production adds a web/tool node later.

## Next (in order)

1. Land `crag_pipeline.py` + `crag_eval.py`, run on Mac against pgvector.
2. A/B: MAX_STEPS=0 (single-shot control) vs 2 (full CRAG) → isolate CRAG's effect
   on refusal + answer quality. This IS the one-knob discipline, automated.
3. If false-refusal too high → tune grader prompt (one knob).
4. THEN self-query metadata routing (Stage-5 shortlist #1) as the next layer.
5. Serving: wrap as an API, add tracing (LangSmith/OTel), latency SLA — Stage 7.
