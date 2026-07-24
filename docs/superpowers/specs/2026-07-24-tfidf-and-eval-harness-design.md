# TF-IDF Restore + Retrieval Eval Harness — Design

_Date: 2026-07-24 · Branch: stage/2_

## Goal

Two independent deliverables:

1. **Re-add TF-IDF** to `RAG/Naive_rag.py` — restore the hand-rolled lexical
   retriever that existed in commit `72a358a` and was dropped in stage 2.
2. **Standalone retrieval eval harness** (`RAG/rag_stage_2/eval/`) — a retriever-agnostic
   runner + gold set that measures retrieval quality over the 7-PDF
   parent-child corpus. Used from here on to tune stages 3–4; not wired into
   `Naive_rag.py`.

Non-goals: LLM-judge / answer-quality metrics, hybrid retrieval, any change to
the dense or parent-child retrieval logic.

## Repository layout — shared root + stage folders

Shared infra lives at `RAG/` root; each stage's own code lives in its own folder.
Stage 2 gets `RAG/rag_stage_2/`; stage 3 will later get `RAG/rag_stage_3/`.

```
RAG/
  generator.py           # SHARED, root: EMBED_MODEL + build_prompt + generate
  Naive_rag.py           # SHARED, root: stage-1 baseline, TF-IDF restored (Deliverable 1)
  docs/                  # SHARED, root: 7 PDFs + RAG_GUIDE.md
  rag_stage_2/
    pdf_extractor.py     # moved; DOCS_DIR -> ../docs
    chunker.py           # moved
    parent_child_rag.py  # moved; imports generator from root
    sections.json        # data (moved)
    chunks.json          # data (moved)
    .dense_cache_pc.npz  # cache (moved)
    eval/
      gold.json          # Deliverable 2
      eval.py            # Deliverable 2
```

- **`generator.py` stays at root** — shared by every stage. The `EMBED_MODEL`
  constant moves **out of `Naive_rag.py` into `generator.py`** so both the
  baseline and stage-2 import it from one shared place.
- **`Naive_rag.py` stays at root** — TF-IDF restored here (Deliverable 1); it
  imports `EMBED_MODEL`/`build_prompt`/`generate` from root `generator.py`. Its
  `DOCS_DIR` is unchanged (`RAG/docs`).
- **Stage-2 files move** into `rag_stage_2/`: `pdf_extractor.py`, `chunker.py`,
  `parent_child_rag.py`, the data JSONs, the `.npz` cache. After the
  `EMBED_MODEL` move, `parent_child_rag.py` no longer imports `Naive_rag` at all
  — it depends only on root `generator.py`.
- **Reaching root `generator.py` from a subfolder:** each stage-2 script inserts
  the repo `RAG/` dir on `sys.path`
  (`sys.path.insert(0, str(Path(__file__).parent.parent))`, and `.parent.parent.parent`
  for `eval/eval.py`) before `import generator` / `import parent_child_rag`.
- **Path edits:** `pdf_extractor.py` `DOCS_DIR -> Path(__file__).parent.parent / "docs"`.
  `chunker.py` reads/writes `sections.json`/`chunks.json` local to `rag_stage_2/`
  (already `__file__`-relative). Cache path stays `__file__`-relative.
- **"First push"** = commit the reorg + new files and `git push origin`.
  Stage-3 work proceeds in a separate `RAG/rag_stage_3/` folder.

---

## Deliverable 1 — TF-IDF in root `RAG/Naive_rag.py`

Faithful restore of the code from `72a358a:RAG/Naive_rag.py`, applied to the
current root `RAG/Naive_rag.py` (which delegates generation to root
`generator.py`).

### Changes
- Move `EMBED_MODEL = "all-MiniLM-L6-v2"` from `Naive_rag.py` into
  `generator.py`; `Naive_rag.py` imports it from `generator` alongside
  `build_prompt`/`generate`.
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
- `python RAG/Naive_rag.py --tfidf --query "what is dense retrieval?"`
  prints a TF-IDF index line and retrieved chunks with descending scores.
- Same script with no flag still runs the dense path unchanged.
- `python RAG/rag_stage_2/parent_child_rag.py --query "..."` still works after
  the `EMBED_MODEL` move (now imported from root `generator`).

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
  - Summary line/table: `N` questions, hit@1, hit@3, hit@5, MRR.
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
- Eval does not alter parent-child **retrieval logic**; it imports
  `parent_child_rag`'s loaders/index and ranks parents itself. (`parent_child_rag.py`
  still gets the reorg's import/`sys.path` edits, but its retrieval behavior is
  unchanged.)
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
