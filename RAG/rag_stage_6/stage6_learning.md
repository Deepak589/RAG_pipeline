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
