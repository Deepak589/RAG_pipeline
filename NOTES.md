# RAG_dev — working notes

Running log of decisions, findings, and open threads. The README is the
canonical roadmap; this file is the reasoning behind the moves.

---

## Where things stand

- Eval harness is live (`RAG/rag_stage_2/eval/`): retrieval-only, framework-free,
  deterministic across runs.
- Reranker (cross-encoder MiniLM-L-12) is done, applied to child chunks then
  collapsed to parents.
- Next up: hybrid retrieval (BM25 + dense) and a chunking rebuild.
- **Open decision before chunking work:** pick a target number and the metric to
  gate on (recall@5? recall@window?).

---

## Architecture: the intended pipeline

```
BM25  ─┐
       ├─ RRF fuse ─→ wide pool (40–50) ─→ cross-encoder rerank ─→ top-3 ─→ LLM
dense ─┘
   (hybrid = retrieval stage)              (refinement stage)
```

- Hybrid is the **retrieval** stage; reranking is a separate **refinement** stage
  on top. They compose in sequence — not either/or.
- Fuse BM25 + dense with **RRF** (rank-based, `score = Σ 1/(60 + rank)`). Rank-based
  fusion sidesteps the fact that BM25 scores and cosine live on different scales.
- **Do not** flatten the cross-encoder into the RRF. It is the most accurate signal
  and the most expensive — give it the final say over a small, high-recall pool.
- Retrieve wide, measure the ceiling (recall@50), rerank, **serve narrow** (top-3).
  A wide candidate pool is a measurement, not a serving choice — the reranker's job
  is to kill the noise down to a clean top-k.

---

## Eval harness

- `qa.json` — golden set. Labels at **parent** level (the unit the generator eats).
  Three buckets: `factual` (doc vocabulary), `paraphrase` (same target as a factual
  item, reworded — tests dense > lexical), `negative` (answer absent — retriever
  should score low / generator should refuse).
- `run_eval.py` — Recall@k, Hit@k, MRR, per-type breakdown, negative score-gap.
  numpy + stdlib only. `--selftest` validates labels + unit-tests metrics with no
  model load.
- Retriever contract: `query -> ranked [(parent_id, score)]`, full ranking. Every
  retriever plugs in through this one shape.
- Parent collapse for eval scores **all** children (group by parent, take max child
  score) — no top-8 cap, so the harness measures the true ceiling.

Hygiene rules:
- Freeze the seed, pin the model name in each result file, save every run to
  `results/<retriever>_<date>.json` so stage-to-stage diffs work.
- **Version the golden set.** Never compare an old baseline against a new retriever
  on different question sets — re-baseline on the same version.
- Grow buckets together; keep paraphrase/negative from getting too small to trust.
- Any chunking change voids the cache and every past eval number. Change **one knob
  at a time**, re-baseline after each.

---

## Findings so far

### Retrieval is a ranking problem, not a recall floor
First run (24 Q): recall@5 ≈ 0.57, but the correct parent was almost always
retrieved — just ranked 7–31, not top-5. That's why reranking was the right next
move rather than more aggressive retrieval.

### Reranker: half-win, two red flags (35 Q, same set)

| metric | dense only | +rerank | delta |
|---|---|---|---|
| MRR | 0.522 | 0.528 | ~flat |
| recall@1 | 0.383 | 0.350 | **−0.03 down** |
| recall@3 | 0.517 | 0.633 | +0.12 |
| recall@5 | 0.633 | 0.700 | +0.07 |
| factual recall@5 | 0.636 | 0.773 | +0.14 |
| paraphrase recall@5 | 0.625 | 0.500 | **−0.13 down** |
| negative gap | +0.225 | +7.81 (logit) | huge |

- **Wins:** factual mid-depth (rank-8 answers pulled into top-3), and negatives —
  cross-encoder separates them hard (neg logit ≈ −3, pos ≈ +4.8). Set the refusal
  threshold at **logit 0** (≈ sigmoid 0.5). The old cosine threshold (0.32 gap) is
  dead — the score scale changed.
- **Red flag 1 — recall@1 dropped, MRR flat.** Reranker sometimes demotes a correct
  rank-1 hit. A 120-word child can lose when the answer needs wider context. To do:
  list questions where dense got R@1=1 but rerank got R@1=0.
- **Red flag 2 — paraphrase recall@5 fell.** ms-marco cross-encoder may punish
  synonym rewording that the bi-encoder handled. Small n (8) so partly noise —
  grow the bucket and re-check.
- **Missing:** reranked run dropped recall@20/@50. Put the ceiling gauge back in
  every run — the reranker can only reorder its candidate window.

### Why hybrid is next
The worst misses (`lewis-ret`, `lewis-gen`, rank 31) were exact-term queries
("DPR", "BART") that dense blurs and BM25 nails. BM25 covers dense's blind spot
(names, IDs, rare terms); dense covers BM25's (paraphrase). A richer hybrid pool
may also cure the reranker regressions by giving it the right candidate to promote.

Measure hybrid in three points on the same set: dense only → hybrid only (watch the
recall@50 ceiling move) → hybrid + rerank.

---

## Chunking rebuild (the weakest link)

Parent size varies ~100× — Gao `III RETRIEVAL` ≈ 22k chars vs Asai `3.2` ≈ 164
chars. Both extremes trace to real eval misses: giant parents are poorly
represented by a lone 120-word child; tiny stubs are too thin to win. Word windows
also cut mid-sentence, and a false heading (REALM footnote "3 Note…") created a
junk parent.

Ladder — one at a time, re-eval after each, stop at target:

1. **Fix parent size variance — mandatory.** Split giant sections on the finer
   heading level (3.1, 3.2…); merge tiny stubs into neighbors. Even size → even
   representation. Biggest single jump expected.
2. **Sentence/paragraph boundaries + token sizing — cheap.** Never cut mid-sentence;
   size by tokens (the embed model and cross-encoder think in tokens, 512 limit),
   not words.
3. **Contextual blurb — biggest recall lever.** Prepend a one-sentence LLM-written
   context to each chunk before embedding ("This passage is from REALM §3.2, on the
   knowledge retriever architecture"). Cache it. Do only if steps 1–2 fall short.
4. **Semantic chunking — skip for now.** Only if 1–3 plateau.

Set a target (e.g. recall@5 ≥ 0.85) and a gate metric before starting step 1. Stop
climbing the moment you hit it — every added stage is complexity and latency you
carry forever.

---

## Stage 3: parent-size split shipped — "regression" is a measurement artifact

Ran Move 1 in `rag_stage_3/pdf_extractor.py` (IEEE subhead split + stub merge).
Size variance **fixed**: parents 111 → 125, char_len min 69 → 536, max 22021 → 9972,
**zero tiny stubs**. Structural goal achieved.

Dense-only eval *looked* worse (same 30 Q):

| metric | stage_2 | stage_3 (dense) |
|---|---|---|
| recall@1 | 0.383 | 0.283 |
| recall@5 | 0.633 | 0.567 |
| MRR | 0.522 | 0.421 |
| recall@8 | — | 0.700 |
| recall@50 | — | **0.867** |

recall@50 = 0.867 → the answer is in the pool; retriever is not broken. Three causes,
in order of blame:

1. **Stale golden labels (the real cause) — broke our own rule.** Split the parents
   but reused old `qa.json` ids. 4/30 questions point at ids that changed meaning or
   vanished: `gao-retrieval` gold `Gao#3` was "III RETRIEVAL" (22k), now "A. Retrieval
   Source" → scores **recall@50 = 0.0** (impossible for a real miss = proof of label
   breakage); `vas-optim`/`vas-optim-para` gold `Vaswani#16` ("5.3 Optimizer") and
   `dpr-vs-bm25` gold `Karpukhin#12` both **merged away**. ≈ the entire top-5 drop.
2. **Splitter title bug (real code defect).** In `_split_giant` + `_merge_within_group`:
   when a heading sits right on its first subhead with < STUB_CHARS intro body, the
   "tiny head → forward" merge **discards the real section title and adopts the first
   subhead's**. That's why `Gao#3` lost "III RETRIEVAL"; each `Gao#3#X` carries the
   *next* subhead's title. `Gao#2` kept "II OVERVIEW OF RAG" only because it had 978ch
   of intro. Second defect: `SUBHEAD_RE` matches `\d+)`, so list items ("1) New
   Modules", a child of "C. Modular RAG") get **promoted to peer parents** — over-split.
3. **Compared dense-vs-dense, skipped rerank.** s3 run is `parent_child_s3` (no rerank).
   recall@8 = 0.70 is exactly the reranker's window; rerank historically pulls mid-depth
   answers into top-3. Judged the split on the wrong pipeline.

Fix order (one knob, re-baseline each):
1. Fix splitter title bug — group head keeps its own heading even when it merges
   forward; restrict the subhead tier to letters `A–Z` (no `\d+)` peer splits).
2. Re-label `qa.json` against corrected stage_3 sections; multi-label broad ones
   (`gao-retrieval` → all of III's sub-parents), re-point merged ones. **Bump version.**
3. Re-run dense-s3 on new labels → real baseline.
4. Then run rerank on s3.
5. Compare only within same label version — never against old stage_2 numbers again.

Lesson (re-learned the hard way): changing chunking changes `section_idx` → **every
gold `parent_id` is invalidated.** Re-label before reading any post-split eval number.

---

## Stage 3: hybrid shipped — it wins, reranker dropped

Two moves since the split: grew the golden set and added hybrid retrieval.

**Golden set v2.** `qa.json` grown 35 -> 55 (factual 22->36, paraphrase 8->12,
negative 5->7) and stamped `"version": "v2-55q-2026-08-02"`. Harness now writes
`qa_version` into every result file (`save_results`), so runs are pinned to a
label version. All old 35-Q baselines are void — everything below is re-baselined
on v2. Caveat still open: label-validation only proves ids *exist*, not that the
20 new questions were labeled against stage-3 sections (verify by hand).

**Hybrid = BM25 + dense, RRF-fused** (`rag_stage_3/hybrid.py`, wired as
`--retriever hybrid` / `bm25` / `hybrid_rerank`; `reranker.py` moved to `RAG/`
root). BM25 is pure-numpy Okapi over children, collapsed to parents by max child
score; fuse dense-parent-rank and BM25-parent-rank by RRF `1/(60+rank)`. Rank
fusion because BM25 and cosine scales are incomparable.

v2 scoreboard (48 positives, same labels):

| retriever | R@1 | R@3 | R@5 | R@20 | R@50 | MRR | fact@5 | para@5 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.375 | 0.698 | 0.771 | 0.963 | 0.992 | 0.564 | **0.833** | 0.583 |
| dense | 0.344 | 0.552 | 0.646 | 0.921 | 0.992 | 0.498 | 0.667 | 0.583 |
| rerank | 0.427 | 0.646 | 0.708 | 0.921 | 0.992 | 0.563 | 0.778 | 0.500 |
| **hybrid** | **0.427** | **0.688** | **0.792** | **0.963** | 0.992 | **0.610** | 0.806 | **0.750** |
| hybrid+rerank | 0.417 | 0.688 | 0.750 | 0.963 | 0.992 | 0.574 | 0.806 | 0.583 |

Findings:
- **BM25 alone beats dense AND rerank at R@5** — this corpus is lexical-heavy;
  dense is the weakest link.
- **Hybrid is the winner** — best or tied on R@1, R@5, MRR. The standout is
  paraphrase R@5 0.583 -> **0.750**: complementarity works (dense rescues the
  reworded queries BM25 misses, BM25 rescues the exact-term queries dense buries).
- **Hybrid ties rerank's R@1 (0.427) with NO reranker**, and beats it everywhere
  else. The reranker never added recall (ceiling recall@50 = 0.992 for all — it
  only reorders its pool); its one job was top-1 precision, and hybrid matches it.

**Decision: drop the reranker.** `hybrid_rerank` (CE reranks the hybrid pool)
made hybrid worse on every axis — R@5 0.792->0.750, MRR 0.610->0.574, paraphrase
0.750->0.583. It took two correct paraphrases from rank 2 and buried them:
`realm-train-para` MRR 0.50->0.10, `lewis-models-para` 0.50->0.08. Same root cause
as the earlier demotions: `ms-marco-MiniLM` is lexically biased and punishes
reworded queries — and now it does so even when the hybrid pool hands it the right
child, so it's the MODEL, not candidate supply. Stage-3 serving pipeline is
**hybrid, no rerank** (R@5 0.792, MRR 0.610, para 0.750).

Only reasons to revisit reranking: tighter top-1, or a clean refusal signal (RRF
score is a weak threshold — keep refusal off hybrid). If so, swap ms-marco for a
paraphrase-tolerant CE (BGE-v2-m3 / mxbai), never reuse ms-marco.

Next: hybrid is the baseline to beat. Remaining chunking-ladder steps (token
sizing, contextual blurb) and the deferred splitter title bug are now optional
polish, gated on a target (recall@5 >= 0.85 not yet hit — hybrid is at 0.792).

---

## Stage 3 — CLOSED. Moving to Stage 4.

**Stage-3 result locked:** serving pipeline is **hybrid (BM25 + dense, RRF), no
reranker**. Best v2 numbers: recall@5 0.792, MRR 0.610, paraphrase 0.750, ceiling
recall@50 0.992. This is the baseline every later stage must beat, on golden set
`v2-55q-2026-08-02`.

**Stage 4 = contextual blurb chunking** (the ladder's step 3, "biggest recall
lever"). Goal: close 0.792 -> the 0.85 gate.

- *What:* prepend a one-sentence, LLM-written context blurb to each child BEFORE
  embedding (e.g. "This passage is from Vaswani §5.3 Optimizer, on the Adam
  warmup+decay schedule"), then embed `blurb + "\n" + chunk_text`. Serve the
  original parent to the LLM unchanged — this is an index-time retrieval trick,
  not a content change. Same idea as Anthropic Contextual Retrieval.
- *Why it should work here:* our ceiling is already 0.992, so this is a RANKING
  problem, not coverage. Blurbs make isolated chunks self-situating — fixes the
  tiny-fragment case (the 43-word Vaswani formula child that embeds to "some
  math") and the near-duplicate-across-papers case. Helps both sides of hybrid:
  better dense vectors AND real keywords for BM25.
- *Cost:* one LLM call per child (~500), CACHED and deterministic — generate
  once, re-embed free. The cache is a COST optimization only; the recall win comes
  from the blurb text, not the cache. (Semantic caching is a different, query-time
  serving concern — irrelevant to this eval, skip for now.)
- *Discipline:* one knob. Build stage-4 chunker with blurbs, re-embed, re-run the
  FULL v2 eval (hybrid, same labels), keep only if recall@5 actually moves. Do NOT
  compare against stage-3 on different chunks without re-checking labels — blurbs
  don't change section_idx, but any parent re-split would.

Open threads carried in (not blockers): verify the 20 new v2 questions were
labeled against stage-3 sections; deferred splitter title bug; refusal signal
still unsolved (hybrid RRF score is a weak threshold).

---

## Housekeeping to-dos

- Clean `sections.json` false headings (regex ate a footnote line as a heading).
- Gitignore the junk currently committed: `*.npz` caches, `__pycache__/`, and decide
  on the PDFs (git-lfs or drop).
- Stages are folders (`rag_stage_2`, `rag_stage_3`), not branches — fine for a
  learning ladder, but retriever fixes must be applied in every copy.
