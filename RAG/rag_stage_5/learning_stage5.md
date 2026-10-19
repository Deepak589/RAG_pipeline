# Stage 5 learnings

## `k=n` (full corpus) on both retrievers kills eval perf at scale

`build_hybrid()` originally set `search_kwargs={"k": n}` (n = 497563, full
child count) on **both** the BM25 and pgvector dense retrievers, so that RRF
fusion wouldn't drop candidates before the `recall@50` cutoff.

Why it was slow (70+ min, still not done):

- **BM25 build** (`BM25Retriever.from_documents`) always tokenizes/indexes
  the whole corpus once, unavoidable — that's a one-time cost regardless of
  `k`, not the bug.
- **BM25 per-query**: `k` doesn't change the scan — rank_bm25 always computes
  a score for all 497k docs per query (vectorized array). `k` only trims how
  many of those already-computed scores get sorted/returned.
- **pgvector per-query**: same shape of mistake — Postgres still computes
  distance to all 497k rows regardless of `k` (no ANN index configured here).
  `k=n` means it also **returns and materializes 497k LangChain `Document`
  objects per query** — that materialization/transfer cost is the real,
  avoidable tax, repeated across 178 questions.

Fix: fetch depth only needs headroom above the metric's max k (`recall@50`),
not the entire corpus. Changed to `k = max(200, max(CEILING_KS))` — RRF still
sees enough candidates from each retriever to not lose anything realistic
before the top-50 cutoff, but pgvector/BM25 stop returning/materializing the
full 497k-row result set on every query.

**Key distinction to remember:** raising/lowering `k` changes what's
*returned*, not what's *scanned*. Both BM25 and exact pgvector search are
inherently O(corpus) per query no matter what — `k` is a return-size knob,
not a scan-size knob. The perf win here comes from cutting downstream
object-construction/transfer cost, not from cutting search cost.

## Hybrid = two independent full-corpus passes, fused after

BM25 (lexical, no embeddings) and dense (embeddings, pgvector) run
independently over the query, each producing their own ranked list, then
`EnsembleRetriever` fuses by rank via RRF (`1/(60+rank)` summed per
retriever). Neither depends on or reranks the other's output — it's parallel
signals, not a pipeline stage. BM25 needs raw text (`chunks.json`), not
embeddings, because it's counting term overlap/IDF, not vector distance —
that's why it can't reuse the pgvector store.

## Paraphrase queries need more dense weight than factual queries

`build_hybrid(weights=(bm25, dense))` defaulted to 0.5/0.5 — no CLI flag
existed to change it, so it never got tuned. Added `--weights <dense> <bm25>`.

Paraphrase-type questions reword the source text (synonyms, restructured
phrasing), which breaks BM25's lexical-overlap signal specifically — the
correct chunk drops in BM25's rank even though dense (bge-m3) still ranks it
high on meaning alone. Under RRF, a doc's fused score is the *sum of per-
retriever rank scores*, so if BM25 ranks it low/absent, that pulls the fused
rank down even when dense alone would've surfaced it.

Reweighting toward dense (0.7/0.3) let dense's better rank dominate the sum,
closing most of the gap: ALL_POSITIVE R@5 0.803→0.830 (now matches the
hand-rolled 0.833 baseline), paraphrase R@5 0.667→0.692. Gain wasn't
paraphrase-only though — factual also rose (0.855→0.882), so it's a general
"trust dense more" tune, not a paraphrase-specific fix. Also note paraphrase
R@50 slipped slightly (0.897→0.872) at 0.7/0.3 — some paraphrase misses are
genuine embedding-similarity misses, not just a fusion-weight artifact, so
weight tuning alone won't fully close that group's gap.
