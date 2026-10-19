# Stage 6 learnings — scaling retrieval (ANN + inverted index)

## Problem

Corpus is 497,563 children / 87,499 parent sections. Neither retrieval path
uses a real index:

- **Dense**: `langchain_postgres.PGVector` — no `CREATE INDEX ... USING hnsw/ivfflat`
  anywhere in `ingest.py` / `lc_pipeline.py`. Brute-force exact NN scan over
  the pgvector column.
- **Sparse**: `BM25Retriever.from_documents` (rank_bm25) — pure-Python,
  in-memory, scores every doc at query time. Not a real inverted index.

Both scale linearly with corpus size → real latency at 500K+ rows, on top of
the CRAG loop's own Claude calls (grade/refine/generate).

## Decision: dense side

**HNSW** on the pgvector column. Chosen over IVFFlat — better recall/speed
tradeoff, no training-pass tuning (`lists` param) needed, standard default
for pgvector today. Tradeoff accepted: slightly higher build time/memory
than IVFFlat.

## Decision: sparse side — deferred, scenario-dependent

Two options weighed:

### Option A — Postgres full-text search (tsvector + GIN)
- Add `tsvector` column/generated column, GIN index, rank via `ts_rank_cd`.
- **Pros**: zero new infra (same DB as pgvector), real inverted index, fast
  at scale, easy to fuse dense+sparse in one SQL query later.
- **Cons**: `ts_rank_cd` is TF/proximity-based, not true BM25 (no k1/b
  tuning) — relevance may shift vs current rank_bm25 results, must re-run
  `crag_eval.py` after switching to confirm no regression. Language config
  (stemming dict) needs to match corpus; current stage5 tokenizer was
  already hand-fitted to match BM25Retriever's tokenizer (see stage5 notes)
  so this is a deliberate departure from that.

### Option B — Dedicated search engine (Elasticsearch/OpenSearch/Meilisearch)
- Separate service, own ingest path, fuse results with dense side via RRF
  in app code (same pattern as now).
- **Pros**: true tunable BM25, mature relevance features (fuzzy/synonyms/
  highlighting), decoupled scaling, per-tenant index isolation, live
  upsert/delete without schema migration — fits multi-tenant dynamic
  uploads and switching between backends.
- **Cons**: new service to deploy/monitor, corpus sync problem (`ingest.py`
  must write both Postgres rows and search-engine docs, handle delete/
  update consistency) — none of that exists today.

**Call**: corpus is currently static, single-tenant, single DB → Option A
is the proportionate fix (ships in ~1 day, no new infra). Revisit Option B
only when dynamic multi-tenant uploads / multi-DB switching becomes a real
near-term requirement — don't build the sync pipeline speculatively.

## Open items

- [ ] Add `CREATE INDEX ... USING hnsw` to pgvector setup (ingest.py or a
      migration step) for the dense collection.
- [ ] If/when switching sparse to Option A: add tsvector column + GIN index,
      rewrite `rank_hybrid`'s BM25 half to query Postgres instead of
      `rank_bm25`, re-run `crag_eval.py` (max-steps 2 vs 0) to confirm no
      relevance regression vs current baseline.
- [ ] Revisit Option B if multi-tenant dynamic ingestion becomes a
      requirement.

## Revisit: sparse side (2026-10-18)

**Why BM25 was dropped:** scale, not relevance. `BM25Retriever` rebuilt over
497K children every process start + scored every doc per query. Option A
(`ts_rank_cd`) fixed speed but its known con ("not true BM25") turned out
decisive:
- `plain` (plainto_tsquery, ANDs every word) matched 0/141 positives → sparse dead.
- `or` recovers 3/34 dense retrieval misses; real IDF BM25 recovers 11/34.

Retrieval-only, 40pct corpus, 279 v4 positives (free, no LLM):

| variant            | hit@5 | hit@20 | misses recovered@5 |
|--------------------|-------|--------|--------------------|
| dense only         | 85.7  | 90.7   | 0/34               |
| dense + pg OR .7/.3| 84.2  | 91.8   | 3/34               |
| dense + bm25 .7/.3 | 86.4  | 91.8   | 7/34               |
| dense + bm25 .5/.5 | 87.5  | 93.2   | 11/34 (loses ~6 dense hits) |

**Now:** `--sparse-mode bm25` = in-memory rank_bm25 over `--chunks`
(~4s build at 199K, 0.67s/q incl. dense). Experiment-only — same scaling
problem as before. `--chunks` must match `--collection` (not checked).

**Decision path:**
1. Paid CRAG A/B decides if BM25 is worth having end-to-end.
2. Worth it → ParadeDB `pg_search` (true BM25 inside Postgres: keeps Option A's
   same-DB + inverted index + scale) — re-verify hit@5 ≈ 87.5 free, then delete
   the in-memory path. Alt: bge-m3 sparse weights in pgvector `sparsevec`.
3. Not worth it → drop sparse, dense-only.

**Paid A/B result (2026-10-18, `v4slice_bm25_55.json`, $0.91):** 91-Q slice
(34 v4 retrieval_miss + 39 neg + 8 bm25-losers + 10 correct controls), bm25
.5/.5 vs stored v4_a3 rows. Misses recovered 10/34, losers lost 7/8, negatives
36→36, controls 10→10. Net +3 → ≈ +1 pt end-to-end on 279 positives, inside
Sonnet-5 run noise. BM25 *swaps* which questions fail, doesn't add many.
**Decision: BM25 dropped, no further sparse testing.** No real end-to-end
gain → no ParadeDB. `--sparse-mode bm25` code removed; baseline stays
`plain` (de-facto dense-only, 0/141 sparse matches).
