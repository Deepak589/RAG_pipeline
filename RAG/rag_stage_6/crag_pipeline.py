#!/usr/bin/env python3
"""Stage 6 — Corrective RAG (CRAG) as a LangGraph state machine.

Layers an agentic retrieve -> GRADE -> refine/re-retrieve -> generate|refuse loop
on top of the Stage-5 hybrid retriever (lc_pipeline.build_hybrid). This is where
the refusal signal (open since Stage 3) finally becomes measurable: a hard-fail
grade with the retry budget spent -> REFUSE instead of hallucinate.

Design contract (keeps the one-knob discipline as an A/B, not a leap of faith):
    MAX_STEPS = 0  -> collapses to single-shot hybrid == the Stage-5 baseline.
    MAX_STEPS = 2  -> full CRAG. The delta between the two IS the measured effect.

The ONLY new LLM calls vs Stage 5 are grade + refine. Single-shot stays the fast
path. Grading uses structured output (enforced JSON) — the small-model
tool-calling gotcha from the notes; with Claude this is reliable.

Deps (run on your Mac):
    pip install langgraph langchain-anthropic \
                langchain langchain-community langchain-huggingface \
                langchain-postgres rank_bm25 sentence-transformers
    export ANTHROPIC_API_KEY=...
    export PG_CONN='postgresql+psycopg://postgres:postgres@localhost:5432/rag'

Run:
    python crag_pipeline.py --ask "Who was the matron of honor at Margaret Brown's wedding?" \
        --vector-store pgvector --collection rag_stage5__bge_m3 --embeddings hf \
        --embed-model BAAI/bge-m3
    python crag_pipeline.py            # interactive REPL (same flags)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# Stage-5 lives in the sibling folder — put it on the path so we can reuse its
# building blocks verbatim (same retriever, tokenizer, parent-collapse contract).
# CRAG must not silently re-tune retrieval; it imports, never re-implements.
S5_DIR = Path(__file__).resolve().parent.parent / "rag_stage_5"
sys.path.insert(0, str(S5_DIR))
import lc_pipeline as s5  # noqa: E402

# chunks.json / sections.json also live in rag_stage_5 (built by ingest.py there).
DEFAULT_CHUNKS = S5_DIR / "chunks.json"
DEFAULT_SECTIONS = S5_DIR / "sections.json"

TOP_K_PARENTS = 5          # docs graded per pass
MAX_STEPS = 2              # query-refinement retries before refusing (0 = single-shot)
REFUSAL_TEXT = "I don't have enough information in the corpus to answer that."


# --------------------------------------------------------------- grade schema
# Structured output = the enforced-JSON discipline from the Stage-5 notes.

class DocGrade(BaseModel):
    """Relevance of ONE retrieved document to the question."""
    verdict: Literal["correct", "ambiguous", "incorrect"] = Field(
        description="correct = clearly answers the question; ambiguous = related "
                    "but insufficient/uncertain; incorrect = off-topic / no answer")
    reason: str = Field(description="one short clause justifying the verdict")


class GradeBatch(BaseModel):
    """Grades for the retrieved documents, in the SAME order they were given."""
    grades: list[DocGrade]


class Rewrite(BaseModel):
    refined_query: str = Field(
        description="a rewritten search query more likely to retrieve the answer "
                    "(disambiguate, decompose, or add domain terms)")


# ------------------------------------------------------------------- graph state

class CRAGState(TypedDict, total=False):
    query: str                       # original user query (never mutated)
    search_query: str                # what retrieve() actually uses this pass
    retrieved: list[dict]            # [{parent_id, text, score}]
    kept: list[dict]                 # docs graded "correct"
    grades: list[dict]               # audit trail of every grade
    steps: int                       # refinement passes taken
    answer: str
    refused: bool
    trace: list[str]                 # node visit order, for eval/debug


# ------------------------------------------------------------------- node factory

def build_crag_app(ensemble, parents, grader_llm, gen_llm,
                   top_k=TOP_K_PARENTS, max_steps=MAX_STEPS):
    """Compile the CRAG graph. `ensemble` + `parents` come from lc_pipeline;
    `grader_llm`/`gen_llm` are ChatAnthropic instances (grader uses structured
    output). Returns a compiled LangGraph app with `.invoke({...})`."""
    from langgraph.graph import StateGraph, START, END

    grader = grader_llm.with_structured_output(GradeBatch)
    rewriter = grader_llm.with_structured_output(Rewrite)

    # -- nodes ---------------------------------------------------------------
    def retrieve(state: CRAGState) -> CRAGState:
        q = state.get("search_query") or state["query"]
        ids, scores = s5.rank_parents(ensemble, q)
        top = ids[:top_k]
        retrieved = [
            {"parent_id": pid, "score": sc,
             "text": parents[pid]["text"], "source": parents[pid]["source"]}
            for pid, sc in zip(top, scores[:top_k])
        ]
        return {"retrieved": retrieved,
                "trace": state.get("trace", []) + ["retrieve"]}

    def grade(state: CRAGState) -> CRAGState:
        docs = state["retrieved"]
        if not docs:
            return {"kept": [], "grades": [], "trace": state["trace"] + ["grade"]}
        # one batched call grades all K docs — cheaper than K calls.
        listing = "\n\n".join(
            f"[Doc {i}] source={d['source']}\n{d['text'][:1200]}"
            for i, d in enumerate(docs))
        prompt = (
            "You are a strict retrieval grader. For EACH document decide whether "
            "it answers the question. Return grades in the SAME order.\n\n"
            f"Question: {state['query']}\n\n{listing}")
        batch = grader.invoke(prompt)
        grades = batch.grades[:len(docs)]
        # KNOB (2026-08-23): keep "ambiguous" too, not just "correct". The eval
        # showed 16 refusals where the GOLD doc was retrieved but the grader
        # graded it below "correct" -> kept nothing -> refined -> refused. The
        # grader's "correct" bar was too strict; "ambiguous" = related-but-
        # uncertain, which is exactly where borderline gold lands. Let the
        # generator be the final decider (it still refuses in-band if the kept
        # context genuinely lacks the answer). Only "incorrect" is dropped.
        KEEP = {"correct", "ambiguous"}
        kept = [d for d, g in zip(docs, grades) if g.verdict in KEEP]
        audit = [{"parent_id": d["parent_id"], "verdict": g.verdict,
                  "reason": g.reason} for d, g in zip(docs, grades)]
        return {"kept": kept, "grades": state.get("grades", []) + audit,
                "trace": state["trace"] + ["grade"]}

    def refine(state: CRAGState) -> CRAGState:
        rw = rewriter.invoke(
            "The search below did not retrieve a document that answers the "
            "question. Rewrite it into a better search query.\n\n"
            f"Question: {state['query']}\n"
            f"Current query: {state.get('search_query') or state['query']}")
        return {"search_query": rw.refined_query,
                "steps": state.get("steps", 0) + 1,
                "trace": state["trace"] + ["refine"]}

    def generate(state: CRAGState) -> CRAGState:
        context = "\n\n".join(f"[{d['source']}]\n{d['text']}" for d in state["kept"])
        prompt = (
            "Answer the question using ONLY the context. If the context does not "
            "actually contain the answer, reply EXACTLY: "
            f"'{REFUSAL_TEXT}'\n\nContext:\n{context}\n\n"
            f"Question: {state['query']}\nAnswer:")
        out = gen_llm.invoke(prompt).text.strip()
        return {"answer": out, "refused": out == REFUSAL_TEXT,
                "trace": state["trace"] + ["generate"]}

    def refuse(state: CRAGState) -> CRAGState:
        return {"answer": REFUSAL_TEXT, "refused": True,
                "trace": state["trace"] + ["refuse"]}

    # -- routing -------------------------------------------------------------
    def route_after_grade(state: CRAGState) -> str:
        if state["kept"]:
            return "generate"
        if state.get("steps", 0) < max_steps:
            return "refine"
        return "refuse"

    # -- wire ----------------------------------------------------------------
    g = StateGraph(CRAGState)
    for name, fn in (("retrieve", retrieve), ("grade", grade),
                     ("refine", refine), ("generate", generate),
                     ("refuse", refuse)):
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "refine": "refine",
                             "refuse": "refuse"})
    g.add_edge("refine", "retrieve")     # the bounded correction loop
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def run_query(app, query: str) -> dict:
    """Single entry point used by both the REPL and the eval harness."""
    return app.invoke({"query": query, "steps": 0, "trace": [], "grades": []})


# ------------------------------------------------------------------------ wiring

def make_chat_anthropic(model, **kwargs):
    """ChatAnthropic factory that drops temperature for Claude-5-family models
    (sonnet-5/opus-5/fable-5), which reject it as deprecated."""
    if model.split("-", 1)[-1].startswith(("sonnet-5", "opus-5", "fable-5")):
        kwargs.pop("temperature", None)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, **kwargs)


def build_from_args(args):
    """Assemble ensemble + parents + LLMs from CLI args. Shared with crag_eval."""
    children = s5.load_children(args.chunks)
    parents = s5.load_parents(args.sections)
    embeddings = s5.make_embeddings(args.embeddings, args.embed_model)
    ensemble = s5.build_hybrid(
        children, embeddings, rrf_c=args.rrf_c,
        weights=(args.weights[1], args.weights[0]),   # main() flips: first=dense
        vector_store=args.vector_store, conn=args.conn, collection=args.collection,
        ef_search=args.ef_search)
    grader_llm = make_chat_anthropic(args.grader_model, temperature=0)
    gen_llm = make_chat_anthropic(args.gen_model, temperature=0)
    app = build_crag_app(ensemble, parents, grader_llm, gen_llm,
                         top_k=args.top_k, max_steps=args.max_steps)
    return app, parents


def add_common_args(ap):
    ap.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    ap.add_argument("--sections", default=str(DEFAULT_SECTIONS))
    ap.add_argument("--embeddings", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--vector-store", choices=["memory", "pgvector"],
                    default="pgvector")
    ap.add_argument("--collection", default="rag_stage5__bge_m3")
    ap.add_argument("--conn", default=os.environ.get(
        "PG_CONN", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"))
    ap.add_argument("--ef-search", type=int, default=400,
                    help="HNSW ef_search (pgvector); forced >= fetch k. Default 40 "
                         "in pgvector is below k and tanks recall -> false refusals.")
    ap.add_argument("--rrf-c", type=int, default=s5.RRF_C)
    ap.add_argument("--weights", nargs=2, type=float, default=[0.7, 0.3],
                    help="dense weight, bm25 weight (Stage-5 baseline = 0.7 0.3)")
    ap.add_argument("--top-k", type=int, default=TOP_K_PARENTS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS,
                    help="refinement retries; 0 = single-shot control (== Stage 5)")
    # grader runs on every query over K docs -> the cost/latency hotspot; Haiku
    # is fast + near-frontier and plenty for relevance classification. Generation
    # wants quality -> Sonnet.
    ap.add_argument("--grader-model", default="claude-haiku-4-5")
    ap.add_argument("--gen-model", default="claude-sonnet-5")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ask", help="one-shot question")
    args = ap.parse_args()

    app, _ = build_from_args(args)
    print(f"CRAG ready (top_k={args.top_k}, max_steps={args.max_steps}, "
          f"grader={args.grader_model})")

    def show(q):
        r = run_query(app, q)
        print("  path:", " -> ".join(r["trace"]))
        for gr in r.get("grades", []):
            print(f"    [{gr['verdict']:9}] {gr['parent_id']}  ({gr['reason']})")
        tag = "REFUSED" if r.get("refused") else "ANSWER"
        print(f"\n  {tag}: {r['answer']}\n")

    if args.ask:
        show(args.ask)
        return
    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            show(q)


if __name__ == "__main__":
    main()
