# Naive RAG — Design Spec (2026-07-19)

## Goal
Learning exercise: build retrieval-augmented generation from scratch. No RAG frameworks, no sklearn. Understand chunking, TF-IDF embeddings, cosine-similarity retrieval, and prompt augmentation.

## Scope
Single file `RAG/Naive_rag.py` (~150 lines). Dependencies: numpy + Python stdlib only.

## Pipeline
1. **Load** — read every `.txt` in `RAG/docs/`.
2. **Chunk** — split each doc into ~200-word chunks with ~40-word overlap. Track source filename per chunk.
3. **Embed (hand-rolled TF-IDF)**
   - Tokenize: lowercase, alphanumeric word regex.
   - TF = term count / chunk length.
   - IDF = log((1 + N) / (1 + df)) + 1 over chunks.
   - Chunk matrix built with numpy; rows L2-normalized.
4. **Retrieve** — embed query with same vocab/IDF, cosine similarity against all chunks, return top-3 with scores.
5. **Generate** — build prompt: retrieved chunks as context + user question, instruction to answer only from context. POST to local Ollama (`http://localhost:11434/api/generate`, model configurable constant, `stream: false`) via urllib. If Ollama unreachable: print retrieved chunks with scores and a notice, no crash.
6. **CLI** — interactive loop (`quit` exits). Also `--query "..."` one-shot mode for scripted testing.

## Out of scope
Neural embeddings, BM25, vector DB, reranking, multi-turn chat, streaming.

## Verification
- Seed `RAG/docs/` with 3 sample .txt files on distinct topics.
- `--query` with a question answerable only from one doc → top retrieved chunk comes from that doc.
- With Ollama running: answer reflects retrieved context.
- Without Ollama: graceful fallback prints chunks.
