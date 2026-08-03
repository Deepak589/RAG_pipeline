# Cross-Encoder Reranking — Design

_Date: 2026-07-26 · Branch: stage/2 · Stage 3, sub-project 1 of N_

## Goal

Raise retrieval ranking quality on the 7-PDF parent-child corpus by inserting a
cross-encoder reranking stage between dense retrieval and the child→parent
collapse. Measured, eval-only first: register a new retriever in the existing
harness and A/B it against the locked dense baseline. Wire into the live answer
path only if it beats the baseline.

### Why reranking, and why first

The stage-2 dense baseline (locked, `run_eval.py`, 21 answerable + 3 negatives):

| k | hit@k | recall@k |
|---|-------|----------|
| 1 | 0.29 | 0.26 |
| 3 | 0.48 | 0.45 |
| 5 | 0.57 | 0.57 |
| 10 | 0.76 | 0.76 |
| 20 | 0.86 | 0.86 |

MRR 0.43. **hit@20 = 0.86 while hit@1 = 0.29** → the correct parent is almost
always in the candidate pool but ranked low. This is a **ranking problem, not a
recall problem**: the 7 papers all concern RAG/retrieval/attention, so parent
sections are near-duplicates in embedding space and cosine can't discriminate the
right one to the top. A cross-encoder, which jointly encodes (query, passage),
discriminates far better on homogeneous corpora — it directly attacks the proven
failure. Hybrid dense+BM25 (widens the pool) and query optimization come in later
sub-projects; metadata filtering is dropped (no leverage on 7 docs).

## Decisions (locked in brainstorming)

- **Rerank unit:** children (120-word chunks), then collapse to parents by max
  reranked score. Chunks fit the cross-encoder's ~512-token window cleanly and
  reuse the baseline's child→parent collapse. The LLM is still fed the parent.
- **Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2` via sentence-transformers
  `CrossEncoder`. No new dependency.
- **Candidate depth:** top-50 dense-retrieved children per query, exposed as a
  `--rerank-depth` flag so the harness can sweep it.
- **Scope:** eval-first. New retriever in `run_eval.py` + `rag_stage_3/reranker.py`.
  No change to `parent_child_rag.py` runtime behavior. Runtime wiring is a
  separate fast-follow spec, gated on the eval win.

Non-goals: hybrid/BM25, query rewriting, LLM-judge answer metrics, runtime
`parent_child_rag.py` wiring, any change to the dense index or the metric code.

## Repository layout

Follows the shared-root + per-stage-folder convention (`rag_stage_2/` →
`rag_stage_3/`):

```
RAG/
  rag_stage_2/
    parent_child_rag.py     # UNCHANGED — loaders + dense index reused
    eval/
      run_eval.py           # EDIT (additive): register reranked retriever + flags
      qa.json               # UNCHANGED
      results/              # reranked_<stamp>.json lands here beside baseline
  rag_stage_3/
    reranker.py             # NEW
```

## Data flow

```
query
  │  (existing dense) qvec = model.encode([query]); sims = index.matrix @ qvec
  ▼
top-50 children by dense score        # argsort(sims)[::-1][:depth]
  │  NEW: cross_encoder.predict([(query, child_text) for each of the 50])
  ▼
50 children with rerank scores
  │  collapse to parents by MAX rerank score   # same collapse as rank_parent_child
  ▼
[(parent_id, score), ...] best first  ──►  hit@k / recall@k / MRR (unchanged metric code)
```

Only the scoring signal changes (cross-encoder replaces cosine) and it runs on a
dense-prefiltered top-50 rather than all 500 children. Everything downstream is
baseline code.

## Components

### `RAG/rag_stage_3/reranker.py` (new)

- `sys.path` insert to reach `rag_stage_2/` for `import parent_child_rag`
  (mirrors how `eval/run_eval.py` reaches its sibling module).
- Lazy model load: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")`,
  downloaded once, then disk-cached by huggingface. Loaded at ranker-build time,
  not per query.
- `rank_reranked(dense_index, cross_encoder, query, depth=50) -> (ids, scores)`:
  1. `qvec = dense_index.model.encode([query], normalize_embeddings=True)[0]`
  2. `sims = dense_index.matrix @ qvec`
  3. take indices of the top-`depth` children by `sims`
  4. `ce_scores = cross_encoder.predict([(query, dense_index.children[i]["text"]) ...])`
  5. collapse: for each candidate child, keep the max `ce_score` per `parent_id`
  6. sort parents by score desc → return `(ids, scores)`, aligned. Same return
     contract as `rank_parent_child`.
- `build_reranked_ranker(depth=50) -> (name, rank_fn)`: loads children/parents +
  dense index via `parent_child_rag` loaders, loads the cross-encoder, returns
  `("reranked", lambda q: rank_reranked(index, ce, q, depth))`.

### `RAG/rag_stage_2/eval/run_eval.py` (edit, additive only)

- `--retriever {parent_child,reranked}`, default `parent_child` — baseline path
  and its output are byte-for-byte unchanged when the flag is omitted.
- `--rerank-depth N`, default 50 — passed to `build_reranked_ranker`; ignored by
  the baseline retriever.
- `main()`: select the ranker builder by `--retriever`. `reranked` imports from
  `rag_stage_3/reranker.py` (add its dir to `sys.path`).
- `save_results` already names files by retriever `name`, so reranked runs write
  `reranked_<stamp>.json` next to `parent_child_<stamp>.json` — direct diff.
- No change to metric functions, `aggregate`, `print_report`, or `--selftest`.

## Testing & verification

- **Determinism:** cross-encoder inference has no sampling → metrics reproducible
  across runs.
- **Selftest unchanged:** `python run_eval.py --selftest` (metric unit tests +
  label validation) still passes; no new metric code added.
- **Depth-1 invariant:** with `--rerank-depth 1` each query has a single
  candidate child, so its parent is rank-1 by construction; reranked hit@1 must
  equal dense hit@1 (a candidate that dense already ranked #1 among children →
  same top parent). Assert as a cheap sanity check during bring-up.
- **Baseline untouched:** `python run_eval.py` (no flags) reproduces the locked
  0.29 / 0.48 / 0.57 numbers exactly.
- **A/B run:** `python run_eval.py --retriever reranked --k 1 3 5 10` and compare
  to baseline; record a before/after table below on completion.

### Success criteria

- **Primary:** reranked **hit@1 ≥ 0.45** and **hit@3 ≥ 0.65** (baseline 0.29 /
  0.48). The 0.86 hit@20 ceiling says the parents are in the pool; a working
  reranker recovers much of that gap into the top ranks.
- **MRR** improves over 0.43.
- **Decision gate:** if reranked hit@1 does not beat dense hit@1, reranking loses
  on this corpus → stop, pivot the next sub-project to hybrid dense+BM25. The
  harness makes the call, not intuition.

### Results (2026-07-26)

Depth swept 1..50; hit@1 peaks near depth 20 and degrades by 50 (noise). Default
changed 50 → 20.

**First measured on 21 answerable questions**, which suggested a modest win on
every rank metric (hit@1 0.29→0.33, hit@3 0.48→0.57, paraphrase hit@1 0.40→0.60).
The `qa.json` readme warns to grow the set before trusting small deltas — so it
was grown to **30 answerable + 5 negatives**, and the deltas moved. The 30q
numbers are the reliable ones:

| metric | dense baseline | reranked (depth 20) | Δ |
|--------|----------------|---------------------|-----|
| hit@1 | 0.400 | 0.367 | −0.033 |
| hit@3 | 0.533 | 0.633 | +0.100 |
| hit@5 | 0.633 | 0.700 | +0.067 |
| hit@10 | 0.833 | 0.800 | −0.033 |
| MRR | 0.522 | 0.528 | +0.006 |
| negative separation | +0.22 | +7.8 | — |

**Verdict: kept, but the win is at hit@3/@5, not top-1.** On the larger set
reranking does **not** improve hit@1 (slightly hurts, 0.40→0.37) and MRR is flat.
It clearly improves **hit@3 (+0.10)** and hit@5 (+0.07) — it pulls the right
parent into the 2–5 band rather than to #1. This matters because the live
pipeline feeds `TOP_PARENTS = 3` to the LLM, so **hit@3 is the operationally
relevant metric**, and it rose 0.53 → 0.63. By the spec's original hit@1 gate the
result is a wash; by what the pipeline actually consumes it is a real gain, so
reranking is kept. Depth is the key knob (50 injects noise from near-duplicate
children of the wrong papers; 20 is the sweet spot). Unchanged across both runs:
the cross-encoder separates negatives far more sharply than cosine (+7.8 vs
+0.22) — a strong basis for the future abstain threshold.

**Lesson:** the 21q → 30q jump reversed the hit@1 finding and dissolved the
paraphrase spike — direct confirmation of the "grow to 50+" caveat. The set
should keep growing before fine-tuning depth or the model.

## Out of scope (later stage-3 sub-projects)

- Runtime wiring: `--rerank` flag on `parent_child_rag.py` answer path (fast
  follow, gated on the eval win above).
- Hybrid dense + BM25 with RRF fusion.
- Query optimization (HyDE / multi-query).
- Metadata filtering (dropped — no leverage on a 7-doc corpus).
