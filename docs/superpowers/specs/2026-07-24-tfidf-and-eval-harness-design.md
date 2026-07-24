# TF-IDF Restore + Retrieval Eval Harness — Design

_Date: 2026-07-24 · Branch: stage/2_

## Goal

Two independent deliverables:

1. **Re-add TF-IDF** to `RAG/Naive_rag.py` — restore the hand-rolled lexical
   retriever that existed in commit `72a358a` and was dropped in stage 2.
2. **Standalone retrieval eval harness** (`RAG/eval/`) — a retriever-agnostic
   runner + gold set that measures retrieval quality over the 7-PDF
   parent-child corpus. Used from here on to tune stages 3–4; not wired into
   `Naive_rag.py`.

Non-goals: LLM-judge / answer-quality metrics, hybrid retrieval, any change to
the dense or parent-child retrieval logic.

## Repository layout — stage snapshot

All of this work lands in a new **frozen, standalone** folder `RAG/rag_stage_2/`.
Stage 3 will later copy from it into `RAG/rag_stage_3/` and evolve there; each
stage folder runs independently.

```
RAG/rag_stage_2/
  generator.py           # copy of RAG/generator.py
  Naive_rag.py           # copy, with TF-IDF restored (Deliverable 1)
  pdf_extractor.py       # copy; DOCS_DIR -> ../docs
  chunker.py             # copy
  parent_child_rag.py    # moved from RAG/
  sections.json          # data (copy)
  chunks.json            # data (copy)
  .dense_cache_pc.npz    # cache (copy)
  eval/
    gold.json            # Deliverable 2
    eval.py              # Deliverable 2
```

- **Shared modules** (`generator.py`, `Naive_rag.py`) are **copied** in so the
  snapshot is self-contained; the originals remain at `RAG/` root so the
  stage-1 baseline still runs.
- **Stage-2-unique files** (`parent_child_rag.py`, `pdf_extractor.py`,
  `chunker.py`, the data JSONs, the `.npz` cache) are **moved** into the folder.
- Sibling imports (`from Naive_rag import EMBED_MODEL`,
  `from generator import ...`) resolve unchanged because Python puts the script's
  own directory on `sys.path`.
- Path edits required: `DOCS_DIR` in `Naive_rag.py` and `pdf_extractor.py` →
  `Path(__file__).parent.parent / "docs"` (shared PDFs/guide stay in `RAG/docs`).
  `eval/eval.py` inserts its parent dir on `sys.path` to import
  `parent_child_rag`.
- "First push" = commit the new folder and `git push origin`. Stage-3 work
  proceeds in a separate `RAG/rag_stage_3/` folder.

---

## Deliverable 1 — TF-IDF in `rag_stage_2/Naive_rag.py`

Faithful restore of the code from `72a358a:RAG/Naive_rag.py`, applied to the
snapshot copy `RAG/rag_stage_2/Naive_rag.py` (which delegates generation to the
in-folder `generator.py`).

### Changes
- Add `import re`.
- Add `tokenize(text)` → `re.findall(r"[a-z0-9]+", text.lower())`.
- Add `TfidfIndex` class:
  - `TF` = term count / chunk length; `IDF = log((1 + N) / (1 + df)) + 1`.
  - Rows L2-normalized so cosine = dot product.
  - `embed_query(query)` ignores out-of-vocab terms; returns zero vector when
    no query term is in vocab.
  - `retrieve(query, k=TOP_K)` → `[(score, source, text)]`, best first —
    identical signature to `DenseIndex.retrieve`.
- Add `--tfidf` flag in `main()`: default remains dense; `--tfidf` selects
  `TfidfIndex(chunks)` and prints `TF-IDF index: N chunks from M files, vocab=V`.

### Constraints
- Additive only — the dense path, caching, and `answer()` are untouched.
- Operates on the existing `.md` corpus (`load_and_chunk()` → RAG_GUIDE.md), as
  before. TF-IDF is not indexed over the PDFs.
- No new dependencies (numpy + stdlib `re` only).

### Verification
- `python RAG/rag_stage_2/Naive_rag.py --tfidf --query "what is dense retrieval?"`
  prints a TF-IDF index line and retrieved chunks with descending scores.
- Same script with no flag still runs the dense path unchanged.

---

## Deliverable 2 — Eval harness (`RAG/rag_stage_2/eval/`)

Retriever-agnostic retrieval eval over the 7-PDF parent-child pipeline.

### Files

**`RAG/rag_stage_2/eval/gold.json`** — JSON list of objects:
```json
{ "id": "q01", "question": "What loss does DPR train the dual encoders with?",
  "source": "Karpukhin et al. - 2020 - Dense Passage Retrieval for Open-Domain QA.pdf" }
```
- `source` = exact PDF filename string, matching the `source` field in
  `sections.json`. The 7 valid values are the 7 PDFs in `RAG/docs/`.
- Label granularity: **source paper only**. Ranking is at the **parent** level
  (parents are what the LLM is fed as context), but a retrieval is correct when
  the gold `source` appears among the top-k ranked parents' sources.
- Sourcing: I draft ~15 (2–3 per paper) grounded in the actual corpus; the user
  adds their own questions and reviews/trims the set before it is used.

**`RAG/rag_stage_2/eval/eval.py`** — the runner:
- Retriever adapter interface:
  `retriever_fn(query) -> [source, source, ...]` — the **full** ranked list of
  parent source filenames, deduped, order preserved (not capped at 3).
- A wrapper `parent_child_retriever()` builds the index via
  `parent_child_rag.load_index(load_children(), load_parents())` and ranks the
  **entire** parent set: score every child, collapse to unique parents keeping
  each parent's best child score, sort descending → full ranked parent list,
  then map to `[p["source"], ...]`. This is eval-side ranking (does not use
  `retrieve()`'s `TOP_CHILDREN`/`TOP_PARENTS` caps, so k up to 5 is measurable).
  Future stages register a new wrapper (one function); the metric code is shared.
- Metrics (deterministic, no LLM; one relevant paper per question):
  - **hit@k** (= recall@k here) for k = 1, 3, 5.
  - **MRR** = mean of 1 / (rank of first correct source), 0 if absent.
- Output:
  - Summary line/table: `N` questions, hit@1, hit@3, MRR.
  - Per-question pass/fail list: `id · ✓/✗ · rank · question`.

### Data flow
```
gold.json ──► eval.py ──► for each q: retriever_fn(q) ──► ranked sources
                                    └─► compare to gold.source ──► hit@k, MRR
                                                                        │
                                                        summary + per-q table
```

### Constraints
- Eval ranks the full parent set (7 papers, 112 parents), so k = 1, 3, 5 are all
  computable. `retrieve()`'s top-3 cap is a runtime concern, not the eval's.
- No change to `parent_child_rag.py`; eval imports its loaders/index and ranks
  parents itself.
- Naive/TF-IDF retrievers index the `.md` guide, not the PDFs, so they are not
  run by this eval (separate corpus).

### Verification
- `python RAG/rag_stage_2/eval/eval.py` loads gold.json, ranks the full parent set with the
  parent-child retriever, prints the summary + per-question table with plausible
  hit rates rising with k (hit@1 ≤ hit@3 ≤ hit@5).
- Gold `source` values validate against the 7 known PDF filenames at load time
  (fail fast on typo).

---

## Out of scope (future stages)
- LLM-judge answer-quality metrics (faithfulness/correctness).
- Hybrid dense + BM25 retriever (stage 3) — will register as a new
  `retriever_fn` wrapper in `eval.py`.
- Persistence / SQLite / metadata filtering.
