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

---

## Deliverable 1 — TF-IDF in `Naive_rag.py`

Faithful restore of the code from `72a358a:RAG/Naive_rag.py`, adapted to the
current file (which now delegates generation to `generator.py`).

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
- `python RAG/Naive_rag.py --tfidf --query "what is dense retrieval?"` prints a
  TF-IDF index line and retrieved chunks with descending scores.
- `python RAG/Naive_rag.py --query "..."` (no flag) still runs the dense path
  unchanged.

---

## Deliverable 2 — Eval harness (`RAG/eval/`)

Retriever-agnostic retrieval eval over the 7-PDF parent-child pipeline.

### Files

**`RAG/eval/gold.json`** — JSON list of objects:
```json
{ "id": "q01", "question": "What loss does DPR train the dual encoders with?",
  "source": "Karpukhin et al. - 2020 - Dense Passage Retrieval for Open-Domain QA.pdf" }
```
- `source` = exact PDF filename string, matching the `source` field in
  `sections.json`. The 7 valid values are the 7 PDFs in `RAG/docs/`.
- Label granularity: **source paper only**. A retrieval is correct when the
  gold `source` appears among the retrieved parents' sources.
- Sourcing: I draft ~15 (2–3 per paper) grounded in the actual corpus; the user
  adds their own questions and reviews/trims the set before it is used.

**`RAG/eval/eval.py`** — the runner:
- Retriever adapter interface:
  `retriever_fn(query) -> [source, source, ...]` — the ranked list of parent
  source filenames, deduped, order preserved.
- A wrapper `parent_child_retriever()` builds the index via
  `parent_child_rag.load_index(load_children(), load_parents())` and returns a
  `retriever_fn` that maps `index.retrieve(query)` → `[p["source"] for _, p in ...]`.
  Future stages register a new wrapper (one function); the metric code is shared.
- Metrics (deterministic, no LLM; one relevant paper per question):
  - **hit@k** (= recall@k here) for k = 1 and k = 3.
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
- k capped at 3 (parent-child returns `TOP_PARENTS = 3`). If a retriever returns
  more, larger k is still computable.
- No change to `parent_child_rag.py`; eval imports its loaders/index.
- Naive/TF-IDF retrievers index the `.md` guide, not the PDFs, so they are not
  run by this eval (separate corpus).

### Verification
- `python RAG/eval/eval.py` loads gold.json, runs the parent-child retriever,
  prints the summary + per-question table with plausible hit rates (> random;
  random hit@3 with 3 of 7 papers ≈ 0.43).
- Gold `source` values validate against the 7 known PDF filenames at load time
  (fail fast on typo).

---

## Out of scope (future stages)
- LLM-judge answer-quality metrics (faithfulness/correctness).
- Hybrid dense + BM25 retriever (stage 3) — will register as a new
  `retriever_fn` wrapper in `eval.py`.
- Persistence / SQLite / metadata filtering.
