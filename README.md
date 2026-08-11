# RAG_dev

Retrieval-Augmented Generation built from scratch, one measurable stage at a
time — no metric moved forward without an eval number behind it.

Stages 1–4 are framework-free (numpy + stdlib only) to keep every mechanic —
chunking, BM25, RRF fusion, cross-encoder reranking, contextual embeddings —
visible and hand-verified. Stage 5 graduates to LangChain + pgvector once the
mechanics were proven, and scales the same pipeline to a ~740-document mixed
corpus (arxiv, SEC filings, Wikipedia, scanned archive PDFs, ReadTheDocs).

**Current best (v2-55q golden set, 48 positives):** hybrid BM25+dense retrieval
over contextually-blurbed chunks — **recall@5 0.833, recall@1 0.531, MRR
0.672**, reproduced on LangChain at **0.854 recall@5** post tokenizer-parity fix.

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
- Chunking changes invalidate every downstream gold label (`parent_id` shifts
  when sections re-split) — re-baselining discipline exists because this bit
  once for real (a "regression" that was actually stale labels, caught and
  root-caused instead of shipped as a retrieval fix).

See [`NOTES.md`](NOTES.md) for the full run log, every scoreboard, and the
reasoning behind each decision.

---

## Pipeline (current — Stage 5)

```
                         ┌─ digital PDF ─→ PyMuPDFLoader
   corpus (~740 docs,    ├─ HTML (SEC) ──→ BSHTMLLoader
   mixed sources)   ─────┼─ scanned PDF ──→ DoclingLoader + OCR
                         └─ markdown ─────→ TextLoader
                                │
                                ▼
                  parent/child structural chunking
              (heading-aware split, size-variance fixed)
                                │
                                ▼
              contextual blurb prepended to each child
             (LLM-written 1-line context, cached, deterministic)
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
              BM25 (lexical)           dense embeddings
           blurb-aware tokenizer      (bi-encoder, pgvector)
                    │                        │
                    └─────────► RRF ◄────────┘
                          (rank fusion, k=60)
                                │
                                ▼
                    top-k parents → LLM answer
                    (Ollama local model, or refuse
                     if retrieval confidence is low)
```

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

Full scoreboards, failure analyses, and the reasoning behind every drop/keep
decision are in [`NOTES.md`](NOTES.md).

---

## Eval harness (the constant across every stage)

- `qa.json` — versioned golden set (`v2-55q-2026-08-02`), labeled at parent
  level, three buckets: `factual`, `paraphrase` (tests dense > lexical),
  `negative` (retriever should score low / generator should refuse).
- `run_eval.py` — recall@k, Hit@k, MRR, per-bucket breakdown, negative
  score-gap. Every result file is stamped with its `qa_version` and model
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
    ingest.py                file-type-routed parsing → sections/chunks + pgvector upsert
    lc_pipeline.py            LangChain retrieval pipeline (EnsembleRetriever)
    qa_gen.py                 synthetic QA generation for the new corpus
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

# eval any retriever against the golden set
python rag_stage_2/eval/run_eval.py --retriever hybrid --stage4
```

Without Ollama running, retrieval still works — chunks are returned raw
instead of an LLM answer.

---

## Roadmap — what's next (Stage 5, in progress)

Reframed as three axes that partly conflict — **scale** (infra: vector DB, ANN
index), **real-time** (systems: caching, batching), **agentic** (reasoning
quality, but *adds* latency/cost) — so agentic work is sequenced last and
gated behind a cheap "does this query need it" router, not applied to every
query.

1. **Golden set for the new 740-doc corpus** — current blocker. New chunking
   voids every old `parent_id`; `qa_gen.py` drafts synthetic labels, human
   spot-check required before trusting `--eval` on the new corpus.
2. **Verify parser quality per source** (scanned PDFs via OCR, SEC HTML
   tables) before trusting any recall number on the new corpus.
3. **Move BM25 into Postgres full-text search** — kills the in-process
   `rank_bm25` bottleneck, gets both retrieval sides living in the same store.
4. **Re-tune chunking** on Docling's structure output — current ingest uses
   blind char windows, a regression from the stage-3 heading-aware splitter.
5. **Reranking returns at scale** — bigger candidate pool makes top-k
   precision matter again. Paraphrase-tolerant model only.
6. **Agentic, gated and last** — CRAG-style retrieve → grade → re-retrieve/
   refuse (closes the refusal-signal gap open since stage 3), then self-query
   metadata routing (`source_type=sec`) via pgvector's jsonb filter instead of
   a hand-rolled router. Single-shot hybrid stays the default fast path.
7. **Latency last, against an SLA** — HNSW/ANN, semantic cache, async batched
   embedding, once a p95 target actually exists to design backward from.

Shortlisted (not started) beyond dense+BM25 hybrid, ranked by expected
leverage if the current pipeline plateaus: self-query metadata retriever,
SPLADE (learned sparse, targets paraphrase), multi-query/HyDE fan-out,
RAPTOR (hierarchical summarization, for long-doc questions only), ColBERT
late-interaction (only if reranker latency becomes the bottleneck).

## Non-goals

No production serving layer, no auth, no multi-turn chat memory, no
streaming. This is a repo for proving retrieval-quality decisions with
numbers, not for operating a deployed RAG service.
