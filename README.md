# RAG_dev

Retrieval-Augmented Generation built from scratch, as a learning ladder.

The rule for this repo: **no RAG frameworks.** No LangChain, no LlamaIndex, no
sklearn, no vector database. Every stage — chunking, embedding, similarity
search, prompt assembly — is written out so the mechanics stay visible. The only
heavy dependencies are numpy and a local embedding model.

Stage 1 (naive RAG) is working. Later stages build on the same single-file
implementation rather than replacing it.

---

## Current state — Stage 1: Naive RAG

`RAG/Naive_rag.py`, ~270 lines. One file, no package structure.

```
RAG/docs/*.md
     │
     ├─ split on '## ' headings          _md_sections()
     ├─ 200-word windows, 40-word overlap _word_chunks()
     ├─ prepend section heading to each chunk
     └─ skip Table of Contents sections   load_and_chunk()
     │
     ▼
  35 chunks
     │
     ├─ dense: all-MiniLM-L6-v2, 384-dim, L2-normalized   DenseIndex   (default)
     └─ sparse: hand-rolled TF-IDF, numpy only            TfidfIndex   (--tfidf)
     │
     ▼
  query embedded with same model → cosine similarity → top-3
     │
     ▼
  prompt: retrieved chunks as context + "answer ONLY from context"
     │
     ▼
  local Ollama (qwen3.5) → answer
  Ollama down/slow → print retrieved chunks instead, no crash
```

### Two retrievers, on purpose

Both are kept so the difference is measurable rather than assumed:

| | TF-IDF (`--tfidf`) | Dense (default) |
|---|---|---|
| Match type | exact term overlap | semantic / paraphrase |
| Built from | numpy + stdlib, hand-rolled | `sentence-transformers` bi-encoder |
| Fails on | synonyms, rephrasing | rare exact terms, names, IDs |

### Retrieval-quality fixes already applied

- **Heading prefix** — every chunk carries its `## Section` heading, not just the
  first window of a section. Without it, later windows lose their topic and
  retrieve poorly on topical queries.
- **Table-of-Contents filtering** — the ToC section lists all 14 headings, so it
  partially matched *any* query and crowded out the section that actually held
  the answer. It's skipped at index time.
- **Fingerprinted vector cache** — `.dense_cache.npz` stores chunk embeddings
  keyed by a SHA-256 of chunk texts + model name. Change the chunking or the
  model and the cache invalidates itself. Gitignored; regenerates on first run.

---

## Quickstart

```bash
pip install numpy sentence-transformers

# optional, for generated answers rather than raw chunks
ollama serve
ollama pull qwen3.5
```

```bash
cd RAG

python Naive_rag.py --query "how do I split documents into chunks?"  # one-shot
python Naive_rag.py                                                  # interactive REPL
python Naive_rag.py --tfidf --query "..."                            # sparse retriever
```

Without Ollama running, retrieval still works — the script prints the retrieved
chunks and the reason generation was skipped.

## Configuration

Constants at the top of `RAG/Naive_rag.py`:

| Constant | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `200` | words per chunk |
| `CHUNK_OVERLAP` | `40` | words shared between neighbors |
| `TOP_K` | `3` | chunks passed to the LLM |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | local bi-encoder, 384 dims |
| `OLLAMA_MODEL` | `qwen3.5` | generation model |

## Layout

```
RAG/
  Naive_rag.py        the whole pipeline
  docs/RAG_GUIDE.md   the corpus — 14 sections on RAG theory
  RAG_GUIDE.docx      same guide, source format
docs/superpowers/specs/
  2026-07-19-naive-rag-design.md   original design spec (stage 1)
```

`RAG_GUIDE.md` does double duty: it's the theory reference *and* the corpus the
pipeline retrieves over. Every technique on the roadmap below is described in it.

---

## Roadmap

Each stage stays framework-free and gets verified against the previous one
before moving on.

- [x] **1. Naive RAG** — fixed chunking, TF-IDF, cosine top-k, context stuffing
- [x] **1b. Dense retrieval** — MiniLM bi-encoder, cached vectors
- [ ] **2. BM25** — proper sparse ranking, replacing raw TF-IDF
- [ ] **3. Hybrid retrieval** — fuse sparse + dense scores (RRF)
- [ ] **4. Reranking** — cross-encoder over the top-N candidates
- [ ] **5. Evaluation harness** — a labeled question set, recall@k and MRR, so
      each change above is proven rather than eyeballed
- [ ] **6. Advanced chunking** — semantic / parent-document strategies
- [ ] **7. Agentic RAG** — query rewriting, multi-hop retrieval

Stage 5 is the pivot point: everything before it is judged by spot-checking
queries, everything after it should be judged by numbers.

## Non-goals

Production concerns are deliberately excluded — no serving layer, no auth, no
managed vector DB, no multi-turn chat memory, no streaming. This is a repo for
understanding RAG, not for operating it.
