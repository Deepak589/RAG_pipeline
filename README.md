# RAG_dev

Retrieval-Augmented Generation built from scratch, one measurable stage at a
time — no metric moved forward without an eval number behind it.

Stages 1–4 are framework-free (numpy + stdlib only) to keep every mechanic —
chunking, BM25, RRF fusion, cross-encoder reranking, contextual embeddings —
visible and hand-verified. Stage 5 graduates to LangChain + pgvector once the
mechanics were proven, and scales the same pipeline to a mixed corpus (arxiv,
SEC filings, Wikipedia, scanned archive PDFs, ReadTheDocs) — 87,499 parent
sections / 497,563 children. Stage 6 puts a **CRAG** (corrective RAG) loop on
top: Claude grades every retrieved doc, refines the query and re-retrieves on a
miss, and **refuses** when the corpus genuinely can't answer.

**Current best — retrieval (`qa_stage5_v3`, 160 Q: 102 factual / 39 paraphrase /
19 negative):** hybrid BM25+dense, BGE-M3 on pgvector — **recall@5 0.830**.

**Current best — end-to-end (Stage 6 CRAG, same golden set):**

| metric | value |
|---|---|
| correct refusal (19 negatives) | **100%** |
| false refusal (141 positives) | **22.7%** |
| answer correctness (of answered) | **89.9%** |
| end-to-end correct | **69.5%** |
| latency | **8.1 s/query** |

The refusal metric had been an open hole since Stage 3 — nothing in the
single-shot pipeline ever declined to answer. CRAG closes it at 100/0 on the
negatives without the false-refusal cost running away.

---

## Why this repo is structured the way it is

Every stage is a folder, not a rewrite. Each one:
1. Changes exactly **one knob** (chunking, retrieval, reranking, ...).
2. Re-runs the **same eval harness** against the **same versioned golden set**.
3. Keeps or reverts the change based on the number, not intuition.

This surfaced real, non-obvious findings — not textbook ones:
- A cross-encoder reranker **hurt** recall@1 and paraphrase recall on this
  corpus (ms-marco is lexically biased) — dropped it, contrary to the "always
  rerank" default.
- **BM25 alone beat dense retrieval** on this corpus — it's lexical-heavy
  (paper titles, model names, IDs). Hybrid still won overall, but the
  "dense-first" assumption didn't hold.
- Contextual blurbs (prepending an LLM-written one-line context to each chunk
  before embedding) turned out to help **BM25's recall more than dense's** —
  the opposite of the going-in hypothesis. Only found because both sides were
  eval'd in isolation (`--raw-bm25` vs `--blurb-bm25`).
- Adopting LangChain reproduced the hand-rolled baseline almost exactly
  (0.854 vs 0.833) — but only after catching a **silent framework default**:
  `BM25Retriever`'s default tokenizer doesn't lowercase or strip punctuation,
  quietly costing 0.06 recall with no error raised. Diffing framework defaults
  against the hand-tuned version before trusting them is now a hard rule.
- **Adding an ANN index made answers worse, not just faster.** HNSW cut latency
  20.6 → 9.8 s/q but end-to-end correctness fell 66.0% → 51.8% and false
  refusals nearly doubled (24.1% → 39.7%). Root cause was not the index: pgvector's
  session default `hnsw.ef_search = 40` was **below the fetch `k`**, so the ANN
  search silently returned a short, worse candidate list. Pinning
  `ef_search >= k` (400) on every pooled connection recovered it to 61.7% while
  keeping the speedup. A speed knob that quietly changes recall is the exact
  failure the eval gate exists to catch.
- **The grader prompt is a retrieval knob in disguise.** Same retrieval, same
  index — rewriting the doc-grading prompt (v2) moved end-to-end correctness
  61.7% → **69.5%** and cut false refusals 31.2% → 22.7%, mostly by fixing
  `refused_on_hit` (27 → 14): cases where the right parent *was* retrieved and
  the grader threw it away. In an agentic pipeline the grader, not the
  retriever, became the top loss bucket.
- Chunking changes invalidate every downstream gold label (`parent_id` shifts
  when sections re-split) — re-baselining discipline exists because this bit
  once for real (a "regression" that was actually stale labels, caught and
  root-caused instead of shipped as a retrieval fix).

See [`NOTES.md`](NOTES.md) for the full run log, every scoreboard, and the
reasoning behind each decision.

---

## Pipeline (current — Stage 6)

```
                         ┌─ digital PDF ─→ PyMuPDFLoader
   corpus (87.5k parent  ├─ HTML (SEC) ──→ BSHTMLLoader
   sections, 497k        ┼─ scanned PDF ──→ DoclingLoader + OCR
   children)             └─ markdown ─────→ TextLoader
                                │
                                ▼
                     parent/child chunking
              (char windows — heading-aware splitter and
               contextual blurbs from stage 3/4 not yet
               ported to this corpus; see roadmap #3)
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
          Postgres full-text            dense embeddings
          (tsvector + GIN)          (BGE-M3, pgvector + HNSW)
                    │                        │
                    └─────────► RRF ◄────────┘
                       (rank fusion, c=60, 70/30 dense)
                                │
                                ▼
                    top-K parents (K=5)
                                │
                                ▼
                   ┌────── CRAG loop (Stage 6) ──────┐
                   │  grade   Claude scores each doc │
                   │          correct|ambiguous|bad  │
                   │     ├─ any correct → generate   │
                   │     ├─ none, steps<2 → refine   │
                   │     │      query, re-retrieve   │
                   │     └─ none, budget spent →     │
                   │            REFUSE               │
                   └─────────────────────────────────┘
                                │
                                ▼
                    answer grounded in kept parents only
```

Dense side is HNSW-indexed, sparse side is a Postgres `tsvector` + GIN index —
both retrieval paths are now real indexes in the same store, no in-process
`rank_bm25` scan. `MAX_STEPS=0` collapses CRAG back to single-shot Stage-5
behaviour, which is the A/B control for "did the agentic layer help?".

Grading runs on Claude Haiku 4.5, generation and the eval judge on Claude
Sonnet 5, all at `temperature=0`. Embeddings stay local (BGE-M3) — the embedder
is deliberately held fixed so CRAG's effect is isolated.

Reranking (cross-encoder) is validated and available but **off by default** —
it regressed recall/paraphrase on this corpus at current scale. Slated to come
back at 70k+ children with a paraphrase-tolerant model (BGE-reranker-v2 /
ColBERTv2), never ms-marco.

---

## Stage history

| Stage | What shipped | Result |
|---|---|---|
| 1 | Naive RAG — fixed-window chunking, TF-IDF + dense cosine top-k, Ollama generation | Working baseline, no metrics yet |
| 2 | Retrieval-only eval harness (`recall@k`, `Hit@k`, `MRR`, per-bucket, negative-gap), framework-free, deterministic | Found retrieval was a **ranking** problem, not a recall floor (correct parent almost always in top-31) |
| 3 | Cross-encoder reranking, then hybrid BM25+dense (RRF) | Reranker was a mixed bag (hurt R@1/paraphrase) → **dropped**. Hybrid won outright: R@5 0.792, beat dense+rerank on every axis |
| 4 | Contextual blurb chunking (Anthropic Contextual Retrieval-style), blurb fed to both dense and BM25 | R@5 0.792 → **0.833**, R@1 +0.10. Recall gain traced to the **lexical** side (BM25), not dense — disproved the initial hypothesis with an isolation test |
| 5 | LangChain migration (reproduced stage-4 baseline first), pgvector dense store, file-type-routed ingestion, corpus scaled 7 → ~740 docs | Faithful reproduction at 0.854 R@5 after fixing a framework tokenizer default; infra now scale-ready |
| 6 | CRAG loop (LangGraph state machine: retrieve → grade → refine/refuse), Claude grading + generation, HNSW + tsvector/GIN indexes, automated refusal + answer-quality gate | Refusal closed: **100% correct-refusal / 22.7% false-refusal**, **69.5% end-to-end correct**, 8.1 s/q. Two findings: ANN `ef_search` default silently cost recall; grader prompt was the biggest single lever |

Full scoreboards, failure analyses, and the reasoning behind every drop/keep
decision are in [`NOTES.md`](NOTES.md).

---

## Eval harness (the constant across every stage)

- `qa_stage5_v3.json` — current golden set (160 Q: 102 factual / 39 paraphrase /
  19 negative), labeled at parent level; supersedes `qa.json`
  (`v2-55q-2026-08-02`) which is kept for stage 1–4 comparability. `paraphrase`
  tests dense > lexical; `negative` means the retriever should score low and the
  generator should refuse.
- `run_eval.py` — retrieval-only: recall@k, Hit@k, MRR, per-bucket breakdown,
  negative score-gap.
- `rag_stage_6/crag_eval.py` — end-to-end gate: correct/false refusal rate,
  Claude-as-judge answer correctness, and an **attribution breakdown**
  (`retrieval_miss` / `refused_on_hit` / `answered_wrong` / `correct`) that says
  which stage lost the query. Cost and wall time reported per run, not gated. Every result file is stamped with its `qa_version` and model
  name so stage-to-stage comparisons are never accidentally apples-to-oranges.
- Retriever contract is one shape: `query -> ranked [(parent_id, score)]` —
  every retriever (TF-IDF, dense, BM25, hybrid, hybrid+rerank, LangChain)
  plugs into the same harness.
- `--selftest` validates labels and unit-tests the metrics with no model load.

This harness is deliberately **not** migrated to LangChain — it's the judge,
and framework default drift would silently break comparability.

---

## Layout

```
RAG/
  Naive_rag.py             stage 1 — single-file naive RAG
  reranker.py               cross-encoder reranking (stage 3, off by default)
  rag_stage_2/eval/         eval harness origin (recall@k, MRR, ...)
  rag_stage_3/              PDF section extractor, BM25 + RRF hybrid
  rag_stage_4/              contextual blurb generation (Ollama, cached)
  rag_stage_5/
    ingest.py                file-type-routed parsing → chunks + pgvector upsert, --index builds HNSW + tsvector/GIN
    lc_pipeline.py            LangChain retrieval pipeline (EnsembleRetriever, ef_search pinned)
    pg_search.py              Postgres full-text retriever (tsvector + GIN) — replaces rank_bm25 at scale
    qa_stage5_v3.json         current golden set (160 Q)
    qa_gen.py                 synthetic QA generation for the new corpus
  rag_stage_6/
    crag_pipeline.py          CRAG LangGraph state machine (retrieve → grade → refine/generate/refuse)
    crag_eval.py              refusal + answer-quality gate, Claude-as-judge
    crag_*.json               run results (ef400, grader_v2, indexed, ...)
    stage6_learning.md        ANN / inverted-index scaling decisions
  docs/RAG_GUIDE.md          theory reference doubling as stage-1 corpus
NOTES.md                     full run log — every scoreboard, every decision
```

---

## Quickstart

```bash
pip install -r requirements.txt   # numpy, sentence-transformers, langchain, psycopg[binary], ...

ollama serve
ollama pull qwen3.5

# stage 1 — naive RAG, single file
cd RAG && python Naive_rag.py --query "how do I split documents into chunks?"

# stage 3/4 — hybrid retrieval over blurbed chunks, hand-rolled
python rag_stage_4/hybrid.py --query "..."

# stage 5 — LangChain + pgvector over the full corpus
python rag_stage_5/lc_pipeline.py --vector-store pgvector --ask "..."

# stage 6 — CRAG (agentic) over the same corpus; needs ANTHROPIC_API_KEY
cd RAG/rag_stage_6
python crag_pipeline.py --collection rag_stage5__bge_m3 \
    --embeddings hf --embed-model BAAI/bge-m3 --ask "..."

# eval any retriever against the golden set (retrieval only)
python rag_stage_2/eval/run_eval.py --retriever hybrid --stage4

# end-to-end gate: full CRAG vs the single-shot control
python crag_eval.py --max-steps 2 --out crag_s2.json    # full CRAG
python crag_eval.py --max-steps 0 --out crag_s0.json    # Stage-5 behaviour
```

One-time index build (dense HNSW + sparse tsvector/GIN):

```bash
python rag_stage_5/ingest.py --index
```

Without Ollama running, retrieval still works — chunks are returned raw
instead of an LLM answer.

---

## Roadmap — what's next (Stage 6 shipped, Stage 7 next)

Three axes that partly conflict — **scale** (infra: ANN index, inverted index),
**real-time** (caching, batching, SLA), **agentic** (reasoning quality, but
*adds* latency/cost). Agentic is gated: `MAX_STEPS=0` is still the fast path.

Done in Stage 6:
- ~~BM25 into Postgres full-text search~~ — `pg_search.py`, tsvector + GIN.
- ~~ANN index on the dense side~~ — HNSW, `ef_search` pinned `>= k`.
- ~~CRAG retrieve → grade → re-retrieve/refuse~~ — closes the refusal gap open
  since Stage 3.

Next, in order:
1. **Attribution says where to work.** Current loss buckets on 160 Q:
   `retrieval_miss` 23, `refused_on_hit` 14, `answered_wrong` 6. Retrieval is
   again the top bucket — the grader is no longer the bottleneck.
2. **Self-query metadata routing** — `source_type=sec` etc. via pgvector's jsonb
   filter instead of a hand-rolled router. Targets `retrieval_miss` directly.
3. **Re-tune chunking on Docling's structure output** — current ingest uses blind
   char windows, a regression from the stage-3 heading-aware splitter.
4. **Reranking returns at scale** — bigger candidate pool makes top-k precision
   matter again. Paraphrase-tolerant model only (BGE-reranker-v2 / ColBERTv2),
   never ms-marco.
5. **Embedder A/B, CRAG held fixed** — BGE-M3 vs Voyage vs OpenAI, isolated,
   after the cheaper knobs are exhausted (a re-embed of 497k children is not
   free). Anthropic ships no embedding model; Claude is generation + grading only.
6. **Serving — Stage 7.** API wrapper, tracing (LangSmith/OTel), latency SLA to
   design backward from, semantic cache, async batched embedding.
7. **Web/tool fallback node** — real CRAG re-retrieves from the web on hard-fail;
   here that path is `refuse` because eval is offline. Production adds it.

Shortlisted (not started), ranked by expected leverage if the pipeline plateaus:
SPLADE (learned sparse, targets paraphrase), multi-query/HyDE fan-out, RAPTOR
(hierarchical summarization, long-doc questions only), ColBERT late-interaction
(only if reranker latency becomes the bottleneck).

## Non-goals

No production serving layer, no auth, no multi-turn chat memory, no
streaming. This is a repo for proving retrieval-quality decisions with
numbers, not for operating a deployed RAG service.
