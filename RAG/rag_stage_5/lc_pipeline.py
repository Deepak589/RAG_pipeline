#!/usr/bin/env python3
"""Stage 5 — the same hybrid RAG, rebuilt on LangChain (v1).

Purpose of THIS file: prove the framework reproduces the hand-rolled stage-4
result (R@5 0.833, MRR 0.672 on golden set v2-55q-2026-08-02) on the SAME
corpus and SAME labels BEFORE trusting it on new data. Framework is the only
knob that changes here — same blurbed children, same parents, same qa.json.

Your hand-rolled parts map 1:1 onto LangChain built-ins:
    parent_child_rag.py (retrieve child, serve parent) -> metadata parent_id + collapse
    hybrid.py  BM25 + dense, RRF 1/(60+rank)           -> EnsembleRetriever(c=60)
    reranker.py                                         -> (optional) a rerank retriever
    generator.py Ollama call                            -> ChatOllama
That is ~400 lines of hand-rolled retrieval collapsing to the wiring below.

LangChain v1 note (verified against langchain==1.3): `langchain.retrievers` no
longer exists and `langchain-community` is being sunset. EnsembleRetriever now
lives in `langchain_classic.retrievers`; BM25Retriever still in community.

Deps (run on your Mac):
    pip install langchain langchain-community langchain-ollama \
                langchain-huggingface rank_bm25 sentence-transformers
    ollama serve   # for --ask

Run:
    python lc_pipeline.py --eval                       # reproduce the numbers
    python lc_pipeline.py --ask "what is dense passage retrieval?"
    python lc_pipeline.py                              # interactive REPL
    python lc_pipeline.py --eval --embeddings ollama --embed-model nomic-embed-text
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Point at the existing stage-4 corpus by default (reproduce step). Swap these
# two paths to the new LangChain-ingested corpus once you scale up.
HERE = Path(__file__).resolve().parent
DEFAULT_CHUNKS = HERE.parent / "rag_stage_4" / "chunks.json"
DEFAULT_SECTIONS = HERE.parent / "rag_stage_4" / "sections.json"
DEFAULT_QA = HERE.parent / "eval" / "qa.json"

RRF_C = 60          # EnsembleRetriever RRF constant == hand-rolled hybrid k
TOP_PARENTS = 3
CEILING_KS = (20, 50)

# CRITICAL: match the hand-rolled BM25 tokenizer. LangChain's BM25Retriever
# defaults to `text.split()` — no lowercasing, keeps punctuation — so "BART."
# != "bart" and the exact-term lexical matching (this corpus's biggest recall
# lever) silently degrades. This regex tokenizer reproduces hybrid.py's tokenize().
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text):
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------- loading

def load_children(path):
    return json.loads(Path(path).read_text())


def load_parents(path):
    secs = json.loads(Path(path).read_text())
    return {f"{s['source']}#{s['section_idx']}": s for s in secs}


def make_embeddings(kind, model):
    """Dense embedder. `hf` (default) uses the SAME MiniLM as the hand-rolled
    pipeline so dense is comparable; `ollama` uses a local Ollama embed model."""
    if kind == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    if kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model or "nomic-embed-text")
    raise SystemExit(f"unknown --embeddings {kind}")


# --------------------------------------------------------------- hybrid wiring

def build_hybrid(children, embeddings, rrf_c=RRF_C, weights=(0.5, 0.5),
                 vector_store="memory", conn=None, collection="rag_stage5"):
    """BM25 + dense, RRF-fused — the whole stage-3/4 hybrid in ~6 lines.

    dense side is pluggable:
      memory   — InMemoryVectorStore, embeds on the fly (7-PDF reproduce; no infra)
      pgvector — an EXISTING Postgres+pgvector collection built by ingest.py
                 (persistent, scalable; nothing re-embedded at query time)
    BM25 always indexes the children list (rank_bm25). k is the full child count
    so the fused ranking covers every parent (the harness needs recall@50)."""
    docs = [
        Document(page_content=c["text"],
                 metadata={"parent_id": c["parent_id"], "child_id": c.get("id")})
        for c in children
    ]
    n = len(docs)
    if vector_store == "pgvector":
        from langchain_postgres import PGVector
        store = PGVector(embeddings=embeddings, connection=conn,
                         collection_name=collection, use_jsonb=True)
        dense = store.as_retriever(search_kwargs={"k": n})
    else:
        dense = InMemoryVectorStore.from_documents(docs, embeddings).as_retriever(
            search_kwargs={"k": n})
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenize)
    bm25.k = n
    return EnsembleRetriever(retrievers=[bm25, dense],
                             weights=list(weights), c=rrf_c)


def rank_parents(ensemble, query):
    """Fused child ranking -> unique parent ranking (best child wins its parent).

    Returns (parent_ids, scores) best-first — the harness contract. Score is a
    rank-based proxy (EnsembleRetriever exposes order, not fused magnitudes), so
    recall/hit/MRR are directly comparable to the hand-rolled numbers; the
    negative-gap metric is NOT (it needs a similarity, see the eval note)."""
    docs = ensemble.invoke(query)
    ids, scores, seen = [], [], set()
    for rank, d in enumerate(docs, start=1):
        pid = d.metadata["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
        scores.append(1.0 / rank)
    return ids, scores


# --------------------------------------------------------------------- metrics
# Mirror run_eval.py exactly so numbers are comparable across pipelines.

def recall_at_k(ranked, relevant, k):
    if not relevant:
        return None
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked, relevant, k):
    if not relevant:
        return None
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def mrr(ranked, relevant):
    if not relevant:
        return None
    rel = set(relevant)
    for i, pid in enumerate(ranked, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def gold_ids(q):
    """Stage-4 reuses stage-3 sections, so honor the _s3 label override."""
    return q.get("relevant_parent_ids_s3", q["relevant_parent_ids"])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def run_eval(ensemble, qa_path, ks):
    data = json.loads(Path(qa_path).read_text())
    qs = data["questions"]
    ver = data.get("version", "unversioned")
    rows = []
    for i, q in enumerate(qs, 1):
        ids, _ = rank_parents(ensemble, q["query"])
        rel = gold_ids(q)
        row = {"type": q["type"], "mrr": mrr(ids, rel)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ids, rel, k)
            row[f"hit@{k}"] = hit_at_k(ids, rel, k)
        rows.append(row)
        if i % 10 == 0:
            print(f"  eval {i}/{len(qs)}")

    cols = ["mrr"] + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks]
    pos = [r for r in rows if r["type"] in ("factual", "paraphrase")]
    print("\n" + "=" * 70)
    print(f"LangChain hybrid — qa {ver}   (compare vs hand-rolled R@5 0.833)")
    print("=" * 70)
    hdr = f"{'group':<14}{'n':>4}  " + "".join(f"{c:>12}" for c in cols)
    print(hdr + "\n" + "-" * len(hdr))
    for name, grp in (("ALL_POSITIVE", pos),
                      ("factual", [r for r in rows if r["type"] == "factual"]),
                      ("paraphrase", [r for r in rows if r["type"] == "paraphrase"])):
        if grp:
            print(f"{name:<14}{len(grp):>4}  " +
                  "".join(f"{_mean([r[c] for r in grp]):>12.3f}" for c in cols))
    print("=" * 70)
    print("note: negative score-gap not reported — EnsembleRetriever gives rank, "
          "not similarity. Keep the hand-rolled harness for the refusal metric.")


# --------------------------------------------------------------------- answer

def answer(ensemble, parents, question, llm, top=TOP_PARENTS):
    ids, scores = rank_parents(ensemble, question)
    top_ids = ids[:top]
    print("\nRetrieved parents:")
    for pid, s in zip(top_ids, scores[:top]):
        p = parents[pid]
        print(f"  {s:.3f}  {p['source'][:32]}  §\"{p.get('title','')[:36]}\"")
    # serve ORIGINAL parent text, not the blurbed child text
    context = "\n\n".join(f"[{parents[p]['source']}]\n{parents[p]['text']}"
                          for p in top_ids)
    prompt = ("Answer the question using ONLY the context below. If the context "
              "does not contain the answer, say so.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    try:
        print(f"\nAnswer: {llm.invoke(prompt).content}")
    except Exception as e:
        print(f"\n[LLM unavailable: {str(e)[:80]} — showing context only]")
        for p in top_ids:
            print(f"\n--- {parents[p]['source']} ---\n{parents[p]['text'][:700]}...")


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    ap.add_argument("--sections", default=str(DEFAULT_SECTIONS))
    ap.add_argument("--qa", default=str(DEFAULT_QA))
    ap.add_argument("--eval", action="store_true", help="reproduce the metrics")
    ap.add_argument("--ask", help="one-shot question")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--embeddings", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--llm-model", default="qwen2.5:14b")  # was qwen3.5 (bad tag)
    ap.add_argument("--rrf-c", type=int, default=RRF_C)
    ap.add_argument("--vector-store", choices=["memory", "pgvector"], default="memory",
                    help="dense index: in-memory (reproduce) or pgvector (scaled corpus)")
    ap.add_argument("--conn", default=os.environ.get(
        "PG_CONN", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"))
    ap.add_argument("--collection", default="rag_stage5")
    args = ap.parse_args()

    children = load_children(args.chunks)
    print(f"loaded {len(children)} children from {Path(args.chunks).name} "
          f"(dense={args.vector_store})")
    embeddings = make_embeddings(args.embeddings, args.embed_model)
    ensemble = build_hybrid(children, embeddings, rrf_c=args.rrf_c,
                            vector_store=args.vector_store, conn=args.conn,
                            collection=args.collection)

    if args.eval:
        ks = tuple(sorted(set(args.k) | set(CEILING_KS)))
        run_eval(ensemble, args.qa, ks)
        return

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=args.llm_model, temperature=0)
    parents = load_parents(args.sections)

    if args.ask:
        answer(ensemble, parents, args.ask, llm)
        return
    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            answer(ensemble, parents, q, llm)


if __name__ == "__main__":
    main()
