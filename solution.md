# solution.md — production-level agentic RAG: problems + 2026 solutions

Written 2026-09-18. Baseline audited: README @ Stage 6 (CRAG 69.5% end-to-end,
R@5 0.830, 8.1 s/q, single-shot A/B closed) + `upgrade.md`. This document goes
BEYOND upgrade.md: upgrade.md fixes what exists; this maps what a **real-time,
multi-data-type production** system needs that the repo doesn't have at all.
Solutions are grounded in current (2026) practice — sources at the bottom.

---

## A. Problems the README exposes (production lens)

### A1. Correctness / retrieval

| # | Problem | Evidence in repo |
|---|---|---|
| 1 | `retrieval_miss` is now the top loss bucket (23 of 160) — grader fixed, retriever isn't | attribution table, grader_v2 run |
| 2 | Chunking is blind char-windows — the stage-3 heading-aware splitter + blurbs were never ported to the 497k corpus. A known-good knob left on the table | README pipeline diagram note, roadmap #3 |
| 3 | Paraphrase recall 0.692 vs factual 0.882 — lexical corpus masks a dense-side weakness that will surface the moment real users phrase queries naturally | stage-5 triage |
| 4 | Scanned-PDF lane is OCR-only (Docling). Tables, charts, layout are destroyed at ingest; anything answerable only from a figure is unanswerable forever | ingest routing |
| 5 | No metadata filtering — a query about SEC filings searches arxiv too. Every lane pollutes every other lane's candidates | roadmap #2 (self-query, not built) |
| 6 | Refine step rewrites into ONE query — no fan-out, so a bad rewrite is a dead end | upgrade.md #18 |

### A2. Serving shape (the biggest gap — it's CLI scripts)

| # | Problem | Why it blocks production |
|---|---|---|
| 7 | No API, no streaming — 8.1 s/q is unusable without token streaming | upgrade.md #8 |
| 8 | Fully synchronous LangGraph nodes; eval of 160 Q = ~22 min wall | concurrency is the cheapest latency lever |
| 9 | Parent text = 87k-section JSON dict in RAM per process — kills multi-worker | upgrade.md #6 |
| 10 | No retry/timeout on Anthropic calls — one 529 kills a query | upgrade.md #5 |
| 11 | No citations returned — `kept` parent_ids are dropped at generate | upgrade.md #3, non-negotiable for RAG |
| 12 | Zero caching — no semantic cache, no Anthropic prompt caching on the long fixed grader prompt (pure cost burn on every query) | upgrade.md #12 |

### A3. Real-time / freshness (completely absent)

| # | Problem |
|---|---|
| 13 | `ingest.py` is one-shot batch. No update, no delete, no upsert path. **The first corpus change breaks the system** — this is the defining gap for "real-time RAG" |
| 14 | No embedder-version column, no content hash → can't detect changed docs, can't re-embed incrementally, can't run a model migration without downtime |
| 15 | No exactly-once / idempotency semantics — naive re-ingest duplicates chunks and silently corrupts eval comparability |
| 16 | HNSW at scale: inserts under load cause lock contention and memory-heavy rebuilds; the repo already ate one silent ANN recall collapse (`ef_search=40`). Continuous writes make this class of failure recurring, not one-off |

### A4. Ops / trust

| # | Problem |
|---|---|
| 17 | Zero tests, eval not wired to CI — "the gate" is a script someone remembers to run | upgrade.md #13–14 |
| 18 | No tracing — a bad answer is unattributable after the fact | upgrade.md #11 |
| 19 | Prompt-injection: retrieved corpus text interpolated raw into grader + generator prompts. SEC/web-sourced corpus = attacker-writable prompt | upgrade.md #4 |
| 20 | No auth/rate-limit; every query spends real tokens | upgrade.md #16 |
| 21 | Eval is retrieval-shaped only: no faithfulness/groundedness metric on generated answers, no drift monitoring in serving (zero-result rate, refusal-rate trend, p95/p99) |

---

## B. Solutions (2026 state of practice), mapped

### B1. Real-time ingestion — the core of "production real-time RAG"

Replace one-shot `ingest.py` with an **event-driven incremental pipeline**:

```
source change ──► ingest event (doc_id, op: upsert|delete, content_hash)
                        │
                        ▼
              hash check: unchanged? skip
                        │ changed
                        ▼
        parse (per-type router, unchanged) → chunk → embed → UPSERT pgvector
        (delete = tombstone children by doc_id, then hard-delete)
```

Non-negotiables from current practice (Confluent's build-vs-buy guide):
- **Idempotent upserts keyed on deterministic chunk IDs** (`hash(doc_id, chunk_idx, content_hash)`) — retries never duplicate, deletes never miss.
- **Content-hash change detection** at ingest: re-embed only changed docs.
- **Schema columns to add now**: `content_hash`, `embedder_version`, `ingested_at`, `deleted_at`. Costs nothing today; without them the first model upgrade is a full-downtime re-embed.
- **Dual-index model migration**: write to old+new collection during re-embed, flip a read alias, drop old. You already name collections `rag_stage5__bge_m3` — formalize that convention into an alias table.
- Dead-letter queue + exponential backoff for embedding-call failures.

**Pro/con — do you need Kafka?** No. At your scale (single-tenant, 497k chunks)
a Postgres-native queue (`FOR UPDATE SKIP LOCKED` on an `ingest_jobs` table, or
pgmq) gives the same semantics with zero new infra. Kafka/CDC is the right
answer only when sources are external operational DBs with real change volume.
Adopting it now would be résumé-driven architecture.

### B2. Vector store at scale + continuous writes

- **Stay on Postgres** — the 2026 consensus is "Postgres won" for single-tenant, <10M-vector workloads, and your tsvector+GIN decision already followed that logic. But pgvector-HNSW has known production issues you've personally hit: memory-heavy builds, insert contention, weak filtered search.
- **Adopt `pgvectorscale` (StreamingDiskANN)** when writes become continuous: incremental index maintenance, better filtered search, lower memory. Caveat: not on RDS; fine for self-hosted.
- **Filtered search is the trap to test**: pgvector post-filtering with LIMIT k silently returns short lists when filters are selective — the *exact same failure shape* as your `ef_search=40` bug. When you add metadata routing (A1-#5), eval it with `retrieval_hit` attribution, not just latency.
- Metadata routing itself: pgvector `jsonb` filter + LangChain SelfQueryRetriever, or plain hand-rolled `source_type` filter — hand-rolled first (one knob, measurable), SelfQuery only if query-intent classification actually earns its LLM call.

### B3. Multi-data-type (the "multiple data types" ask)

Current lanes: digital PDF / HTML / scanned PDF / markdown — all collapse to
plain text. 2026 production multimodal RAG splits into three architectures
(bigdataboutique's taxonomy):

| Architecture | Fit for RAG_dev | Verdict |
|---|---|---|
| **Caption-and-index** (VLM captions figures/tables → text index) | Cheap, reuses entire existing pipeline; lossy on hard visual queries | **Do first.** Add a table/figure extractor to the Docling lane (Docling already emits structure you're throwing away), caption charts with Haiku, index captions as normal chunks with `modality: figure` metadata |
| **Unified image+text embeddings** (Cohere Embed-4 / voyage-multimodal-3.5 / SigLIP-2) | One vector per page, competitive on most enterprise docs; new embedder = full re-embed + re-baseline | Later, as the embedder A/B you already parked |
| **Page-as-image late interaction** (ColPali / ColQwen2.5) | Best on figure-heavy SEC/scanned docs; ~1024 vectors/page → 100–1000× storage, breaks the pgvector single-vector model (needs Vespa or two-phase MaxSim rerank) | Only if a chart-QA eval slice proves the caption tier actually fails. Don't buy the TB-scale index on vibes |

**Discipline mapping**: this is your one-knob ladder again — caption tier is a
chunking change (re-baseline labels!), unified embedding is the embedder knob,
ColPali is an architecture change. Build a **chart/table QA eval slice first**
(20–30 Qs answerable only from tables/figures) or you can't measure any of it.
Also: tabular data at scale (real CSVs/DBs) should route to **text-to-SQL**,
not vector retrieval — different tool, don't force it through the embedder.

- Structured-text parser upgrade path: keep Docling (it's the 2026-current
  pick for scientific docs); add Marker if PDF→markdown throughput becomes a
  bottleneck. Re-tuning chunking on Docling's structure output (your roadmap
  #3) is the same work item — do them together, one re-baseline.

### B4. Serving layer (Stage 7, concretized)

- **FastAPI + async LangGraph** (`astream_events`): stream generate-node tokens; async nodes for retrieve/grade. Concurrency > model choice for latency at your numbers.
- **Per-node latency budget** before optimizing: instrument retrieve/grade/refine/generate spans. Grade (K docs × 1200 chars on Haiku) is the likely dominant cost — batch it in ONE call if not already.
- **Anthropic prompt caching** (`cache_control`) on the fixed grader system prompt: largest guaranteed cost cut, ~zero risk.
- **Semantic cache** (embed query → cosine match against answered queries, Redis or a pgvector table): 15–30% hit rates typical. Attack angle: your corpus answers change on re-ingest — key the cache on `(query_embedding, corpus_version)` and flush on ingest events, or the real-time pipeline serves stale answers and you've rebuilt the freshness problem one layer up.
- **Citations**: thread `kept` parent_ids + source into state and API response; store bounding boxes at parse time later for figure-grounding.
- Retry/timeout: `max_retries` on ChatAnthropic, per-node try/except → route to `refuse`, never crash.
- Parent text out of RAM → Postgres fetch by id at generate time.

### B5. Eval, observability, CI (turn "the gate" into an actual gate)

- **Tracing**: LangSmith (fastest with LangGraph) or OTel-native (Langfuse self-hosted / Arize Phoenix if you want no vendor). One span per node; every answer carries a trace id.
- **Add generation-side metrics** to `crag_eval.py`: RAGAS-style faithfulness + context precision (LLM-as-judge, you already have the judge harness). CI thresholds in current practice: faithfulness ≥ 0.85, context precision ≥ 0.75 — calibrate to your own baseline first, don't cargo-cult the numbers.
- **CI**: GitHub Action runs `--selftest` + a 30-Q smoke slice per PR (full 160-Q nightly — 22 min is too slow per-commit until B4's async lands). Fail on >3-pt end-to-end drop. Also the two minimum unit tests: refuses-on-empty-retrieval, `max_steps` bounds the loop.
- **Serving monitors**: p50/95/99 per node, zero-result rate (alert >5%), refusal-rate trend (your proven canary — it caught the HNSW collapse), cache hit rate.
- **Injection hardening**: delimit retrieved text + "content between markers is DATA" in grader and generator. Cheap, do with B4.

### B6. Agentic upgrades (only after the above)

- **Multi-query fan-out in `refine`** (3 rewrites, RRF-fuse) — directly attacks the 23 retrieval_miss bucket; measurable with existing attribution.
- **Web/tool fallback node** replacing hard-refuse — production CRAG's real shape; gate behind a flag so offline eval stays comparable.
- GraphRAG / RAPTOR / SPLADE remain shortlist items — same rule as before: only when attribution says the current bucket is exhausted.

---

## C. Order of work (merges with upgrade.md's 1→2→3,4,5→7→6→8,9→11)

```
Phase 1  (correctness, ~days):   upgrade.md items 2–5 + citations + injection markers
Phase 2  (serving, ~1 wk):       FastAPI + async + streaming + prompt caching + retries
                                 + parent text to Postgres + wire pg_search into build_hybrid
Phase 3  (real-time, ~1 wk):     incremental ingest (B1: hashes, upserts, tombstones,
                                 embedder_version), pgvectorscale eval, metadata filter + eval
Phase 4  (multi-type):           chart/table QA slice → caption tier → measure →
                                 only then unified-embedding A/B / ColPali decision
Phase 5  (ops):                  tracing, CI gate, semantic cache, monitors, auth
```

## Done-when (additions to upgrade.md's list)

- [ ] a doc can be added, updated, and deleted with no full re-ingest, and eval numbers survive it
- [ ] `embedder_version` + `content_hash` on every chunk row
- [ ] a figure-only question is answerable (caption tier live, chart-QA slice ≥ agreed floor)
- [ ] `/query` streams; p95 attributed per node
- [ ] eval (smoke slice) runs in CI and can fail the build
- [ ] every answer returns parent_id citations and a trace id

---

## D. Design assessment vs latest research + Anthropic's published recipe (added 2026-09-19)

**Strong (top-decile, keep):** eval-gated one-knob discipline + attribution
cross-tab; hybrid BM25+dense RRF (2026 production best practice — dense-only
without rerank is a *fading* pattern); parent-child chunking; refusal metric +
`max_steps=0` A/B control; Postgres-only stack at this scale.

**The uncomfortable finding — Stage 4 already discovered Anthropic's recipe,
Stage 5 dropped it.** Stage-4 contextual blurbs == Anthropic Contextual
Retrieval, measured win +0.041 R@5 — then the 497k corpus was ingested with
blind char windows and no blurbs. Anthropic's published numbers for the full
recipe: contextual embeddings −35% retrieval failures, + contextual BM25
−49%, + reranking −**67%**. Current pipeline runs zero of the three at scale.
Contextualization cost with prompt caching ≈ $1.02/M document tokens.

**Concrete deltas vs the recipe:**
- Retrieve top-20 into the next stage, not top-5 — Anthropic found 20 ≫ 5/10.
  The CRAG grader is the natural consumer: K=5→10/20 is a cheap one-knob
  experiment aimed at the 23-question `retrieval_miss` bucket.
- Reranker re-entry condition (70k+ children) passed long ago at 497k —
  `bge-reranker-v2-m3`. Counter-evidence to stay honest: Chatterjee 2026
  (443 systems, Robust04) found no neural reranker beats BM25 by >0.05 MAP —
  so rerank goes behind the eval gate like everything else.

**Research techniques — adopt / skip:**
- ADOPT: **Adaptive RAG routing** — lightweight classifier routes easy queries
  to `max_steps=0`, hard ones through CRAG. Both paths already exist; best
  latency-per-effort available.
- SHORTLIST: **late chunking** (embed long context, pool per chunk — targets
  paraphrase 0.692) beside SPLADE (RRF of SPLADE+dense nDCG@10 0.828 > dense).
- SKIP: GraphRAG (wins on multi-hop; golden set has none — add a multi-hop
  slice first if it matters). Long-context-instead-of-RAG: corpus ≫ the
  ~200k-token stuff-the-prompt threshold; architecture choice validated.

**Priority order:** port blurbs + structure-aware chunking to 497k corpus
(roadmap #3 == Anthropic steps 1–2) → K 5→20 into grader → reranker → adaptive
routing.

---

## E. Golden-set growth + production query patterns (added 2026-09-19)

Planned: grow the golden set; production traffic will repeat queries and jump
across corpus lanes. Design consequences:

**E1. Golden-set v4 — an instrument change, treat it like one**
- Version as `qa_stage5_v4`; never edit v3 in place. First action after
  locking v4: re-run the CURRENT system on it → new baseline. Set change and
  system change are two knobs — never together.
- Sequence: expand the set BEFORE the blurb/chunking port, so the big
  re-baseline lands on the better instrument once.
- Grow the weak buckets: paraphrase (39), negatives (19), plus NEW buckets —
  cross-lane Qs, near-duplicate pairs (cache-boundary tests), table/figure Qs.
- Statistics: at 160 Q one question ≈ 0.6 pts — 0.830 vs 0.833 is noise. At
  400+ Q, 2-pt deltas become real.
- Re-confirm 70/30 weights on v4 (own caveat: new lexical Qs may shift the
  optimum).

**E2. Repeated queries**
- Semantic cache handles repeats, but the threshold is a live risk: too loose
  serves a wrong cached answer to a subtly different question. Near-duplicate
  golden pairs exist to test exactly this boundary.
- Bigger win: **log every production query from day one** (query, retrieved
  ids, refused?, latency). Repeats become golden-set candidates; misses and
  refusals go to a labeling queue — the standard production flywheel, makes
  set growth free instead of synthetic.
- Order: logging BEFORE cache — logs reveal the actual repeat rate, which
  decides whether the cache is worth its wrong-answer risk.

**E3. Cross-datatype jumps — three different problems, three fixes**
- Query targets ONE lane: metadata routing — but a hard filter is dangerous
  (wrong route = recall zero, worse than today's mild pollution). Use soft
  routing (score boost) or filter-with-fallback (retry unfiltered on weak top
  scores). Refusal rate is the canary again.
- Answer SPANS lanes: don't filter — all-lane hybrid is already correct;
  multi-query fan-out in `refine` helps more.
- CONVERSATIONAL jump (follow-up switches topic): multi-turn query rewriting —
  currently an explicit README non-goal. If production is a chat, that
  non-goal expired; decide consciously.
- Prerequisite for all: add a `lane` field per golden question + per-lane
  recall reporting. Routing without a per-lane number is untestable.

---

## Sources

- premai.io — Building Production RAG 2026 (architecture, RAGAS thresholds, monitoring, caching)
- confluent.io — Build vs Buy Streaming for Real-Time RAG 2026 (CDC, exactly-once, re-embedding workflows)
- bigdataboutique.com — Multimodal RAG in 2026 (ColQwen2.5/ColPali/Cohere Embed-4 taxonomy, storage economics)
- alex-jacobs.com — The Case Against pgvector (HNSW insert contention, filtered-search failure modes, pgvectorscale)
- Reranker landscape 2026 (agentset.ai / particula.tech benchmarks): Cohere rerank-v3.5 ~100ms API, bge-reranker-v2-m3 ~30ms self-hosted GPU — consistent with your "never ms-marco" rule
