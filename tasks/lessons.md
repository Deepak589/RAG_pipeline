# Lessons

Running log of bugs caught, fixes applied, and problems known but not yet fixed.
One entry per lesson. Newest at the bottom of each section.

---

## Fixed

### L1 — Ollama timeouts caused by reasoning chain-of-thought

**Symptom:** `generate()` hit the request timeout on almost every query. Retrieval
was instant, so the delay was entirely on the model side.

**Root cause:** `qwen3.5` is a reasoning model. By default it emits a long internal
chain-of-thought before the answer, which took minutes for a prompt that should
answer in seconds.

**Fix:** Send `"think": false` in the Ollama payload (`Naive_rag.py:207`).
Answers now return in seconds.

**Rule:** When a local model is slow, check whether it is a reasoning model before
raising the timeout. Raising the timeout hides the cause; disabling the reasoning
pass removes it.

---

### L2 — Timeout and unreachable were reported as the same error

**Symptom:** "Ollama not reachable" printed even when Ollama was running fine and
had simply not finished generating.

**Root cause:** `urllib` wraps a socket timeout inside `URLError`, so a single
`except URLError` branch caught both "nothing is listening" and "listening but slow".

**Fix:** Unwrap `e.reason` and check for a timeout before falling through to the
unreachable message (`Naive_rag.py:213-220`). Two distinct messages now.

**Rule:** Error handling that collapses two different failures into one message
sends you debugging the wrong system. Distinguish them at the point of catch.

---

### L3 — Table of contents outranked every real answer

**Symptom:** For most queries the top retrieved chunk was the document's table of
contents. Both the dense and the sparse retriever did this, so the actual answer
section was pushed out of top-3.

**Root cause:** A table of contents lists every heading in the document. It
therefore shares vocabulary with every query while containing no information that
answers any of them. High lexical and semantic overlap, zero information density.

**Fix:** Skip sections whose heading contains "table of contents" during chunking
(`Naive_rag.py:73-74`). 36 chunks became 35.

**Why at index time, not query time:** Navigation content is never a valid answer
for any query, so there is no case where filtering it costs recall. Filtering it
once during indexing is cheaper and simpler than post-filtering every query.

**Rule:** When two independent retrievers make the same mistake, the bug is in the
chunks, not in the retriever. Having a second retriever is what made this
diagnosable.

---

### L4 — Later chunks in a section lost their topic

**Symptom:** The second and third word-windows of a long section retrieved poorly
even when the section was the correct answer.

**Root cause:** Fixed-size word windows split a section into pieces. Only the first
piece contains the heading. Window 3 of a chunking section might never use the word
"chunking", so neither the embedding nor the term vector carries the topic.

**Fix:** Prefix every window with `## {heading}` unless it already starts with it
(`Naive_rag.py:76-77`).

**Rule:** Chunk boundaries destroy context that the reader recovers from the page
layout but the retriever cannot. Re-inject structural context into every chunk.

---

### L5 — Embedding cache went stale silently

**Symptom (anticipated, caught before it bit):** A cached vector matrix built from
old chunks would be reused after chunking logic changed, producing retrieval
results that did not match the current code.

**Fix:** Fingerprint the cache with `sha256(model_name + every chunk text)` and
store it alongside the matrix (`Naive_rag.py:165-185`). Any change to chunk text or
embedding model invalidates the cache. Chunk texts are separated by a null byte so
that different chunk boundaries cannot hash to the same value.

**Rule:** A cache keyed only by file path is a lie. Key it by the content that
produced it.

---

## Known problems, not yet fixed

### P1 — No evaluation harness *(highest priority)*

There is no query set, no ground-truth relevance labels, and no recall@k or MRR
measurement. Every claim that a change "improved retrieval" is currently based on
eyeballing three results for one query.

**Consequence:** Each later stage (BM25, hybrid RRF, reranking, better chunking) is
a change we cannot verify. Regressions will be invisible.

**Blocking:** everything after stage 2 in a meaningful sense.

---

### P2 — TF-IDF has no document-length normalization

`TfidfIndex` divides term counts by chunk length as its TF, which under-weights
long chunks and has no term-frequency saturation. BM25 fixes both:

```
BM25 = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len / avglen))
```

**Fix path:** Replace the `TfidfIndex` body with BM25, keep the `retrieve()`
signature. `main()` needs no change because the two index classes are duck-typed.

---

### P3 — Sparse and dense retrievers cannot be used together

They are mutually exclusive via the `--tfidf` flag. Dense misses exact tokens
(identifiers, error codes, config keys); sparse misses paraphrase. Neither alone
covers both.

**Fix path:** Hybrid retrieval — run both, fuse the rank lists with Reciprocal Rank
Fusion. Requires P2 first so the sparse leg is worth fusing.

---

### P4 — Fixed top-3 with no score threshold

`TOP_K = 3` always returns exactly three chunks regardless of score. A query with
one good match still drags in two weak ones; a query with six good matches loses
three. Nothing filters on absolute similarity.

**Fix path:** Score threshold, or a cross-encoder reranker over a wider candidate
set (retrieve 20, rerank, keep what clears the bar).

---

### P5 — Short section tails are silently dropped

`_word_chunks` skips any window shorter than 20 words. A section ending in a short
paragraph loses that text entirely — it is never indexed and can never be
retrieved. No warning is printed.

**Fix path:** Merge a short tail into the previous window instead of discarding it.

---

### P6 — Answers carry no citations

`build_prompt` labels each context block with its source filename, but nothing asks
the model to cite which block it used, and nothing verifies that the answer is
grounded in the retrieved text.

**Fix path:** Ask for inline citations in the prompt; later, verify them against the
retrieved set.

---

### P7 — Dense numpy matrices will not scale

Both indexes hold a full dense matrix in memory. TF-IDF is `n_chunks x vocab_size`,
which is mostly zeros. Retrieval is a full matrix multiply plus a full `argsort`,
which is O(n log n) over every chunk.

**Not urgent:** at 35 chunks this is free. It becomes real somewhere around 10k+
chunks, at which point the answer is a sparse matrix for TF-IDF and a vector store
with ANN search for dense.

---

### P8 — Design spec is stale

`docs/superpowers/specs/2026-07-19-naive-rag-design.md` lists neural embeddings as
out of scope and does not mention table-of-contents filtering. It now contradicts
the implementation.

**Fix path:** Update it, or archive it as a historical record of stage 1's original
scope.
