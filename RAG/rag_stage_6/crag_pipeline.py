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
import threading
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
# CLI knobs stamped into every eval summary (one-knob A/B bookkeeping)
KNOBS = ("max_steps", "retry_on_refusal", "top_k", "sparse_mode", "grade_mode",
         "grader_chars")
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


class SetGrade(BaseModel):
    """KNOB A2: one verdict for the retrieved SET, not each doc."""
    sufficient: bool = Field(
        description="true if the documents TOGETHER contain the answer")
    reason: str = Field(description="one short clause justifying the verdict")


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
                   top_k=TOP_K_PARENTS, max_steps=MAX_STEPS,
                   retry_on_refusal=False, grade_mode="doc", grader_chars=4000):
    """Compile the CRAG graph. `ensemble` + `parents` come from lc_pipeline;
    `grader_llm`/`gen_llm` are ChatAnthropic instances (grader uses structured
    output). Returns a compiled LangGraph app with `.invoke({...})`."""
    from langgraph.graph import StateGraph, START, END

    grader = grader_llm.with_structured_output(GradeBatch)
    set_grader = grader_llm.with_structured_output(SetGrade)
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
            f"[Doc {i}] source={d['source']}\n{d['text'][:grader_chars]}"
            for i, d in enumerate(docs))
        if grade_mode == "set":
            # KNOB A2: per-doc grading drops gold that is only partially relevant
            # on its own. Judge the set; if it suffices, the generator sees ALL
            # docs (it already refuses in-band when the answer is truly absent).
            sg = set_grader.invoke(
                "Do these documents, taken TOGETHER, contain the answer to the "
                f"question?\n\nQuestion: {state['query']}\n\n{listing}")
            v = "correct" if sg.sufficient else "incorrect"
            audit = [{"parent_id": d["parent_id"], "verdict": v, "reason": sg.reason}
                     for d in docs]
            return {"kept": docs if sg.sufficient else [],
                    "grades": state.get("grades", []) + audit,
                    "trace": state["trace"] + ["grade"]}
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

    def route_after_generate(state: CRAGState) -> str:
        """KNOB A1 (2026-10-05): an in-band generator refusal is a FAILED pass,
        not a final answer — send it back through the correction loop.

        Why this exists: the 2026-08-23 grader knob (keep "ambiguous", not just
        "correct") made `kept` almost never empty, so route_after_grade nearly
        always picks "generate". The generator then refuses in-band and the
        graph ends. Net effect: the refine loop became unreachable for the
        common failure — 25 of 32 refusals in crag_grader_v2 ended with
        steps_used=0, i.e. the max_steps=2 budget was never spent on them.
        CRAG without a reachable correction step is just Stage-5 hybrid with
        extra LLM calls.

        Off by default: `retry_on_refusal=False` reproduces the measured
        baseline exactly, so this stays a one-knob A/B."""
        if (retry_on_refusal and state.get("refused")
                and state.get("steps", 0) < max_steps):
            return "refine"
        return "end"

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
    g.add_conditional_edges("generate", route_after_generate,
                            {"refine": "refine", "end": END})
    g.add_edge("refuse", END)
    return g.compile()


def run_query(app, query: str) -> dict:
    """Single entry point used by both the REPL and the eval harness."""
    return app.invoke({"query": query, "steps": 0, "trace": [], "grades": []})


# ---------------------------------------------------------------- spend guards
# Every LLM call in stage 6 goes through make_chat_anthropic, so the meter and
# cache live here once instead of in each script (mistakes.md M4/M6).

# $/MTok (input, output), Anthropic list price 2026-10. Unknown model -> priced
# as the most expensive one so the cap errs toward stopping early.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0), "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-fable-5-1": (10.0, 50.0),
}


class BudgetExceeded(RuntimeError):
    pass


class UsageMeter:
    """Process-wide token/$ ledger. Thread-safe (qa_expand uses a pool)."""

    def __init__(self):
        self.max_usd = float("inf")
        self.by_model = {}
        self._lock = threading.Lock()

    @property
    def usd(self):
        return sum(m["usd"] for m in self.by_model.values())

    def check(self):
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.usd:.2f} >= cap ${self.max_usd:.2f}")

    def add(self, model, usage):
        # ponytail: ignores the prompt-caching discount (we send no cache_control),
        # so cost is a slight overestimate if that ever gets turned on
        hit = usage.get("total_cost") == 0      # langchain-core marks LLM-cache hits this way
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        price = next((p for k, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0]))
                      if model.startswith(k)), max(PRICES.values()))
        usd = 0.0 if hit else (tin * price[0] + tout * price[1]) / 1e6
        with self._lock:
            m = self.by_model.setdefault(
                model, {"calls": 0, "cache_hits": 0, "in": 0, "out": 0, "usd": 0.0})
            m["cache_hits" if hit else "calls"] += 1
            if not hit:
                m["in"] += tin
                m["out"] += tout
                m["usd"] += usd

    def report(self):
        return {"usd": round(self.usd, 4),
                "by_model": {k: {**v, "usd": round(v["usd"], 4)}
                             for k, v in self.by_model.items()}}


METER = UsageMeter()


def _meter_tap(model):
    from langchain_core.callbacks import BaseCallbackHandler

    class _Tap(BaseCallbackHandler):
        raise_error = True      # else langchain logs and swallows BudgetExceeded

        def on_chat_model_start(self, serialized, messages, **kw):
            METER.check()       # refuse BEFORE paying, not after

        def on_llm_end(self, response, **kw):
            for gens in response.generations:
                for g in gens:
                    METER.add(model, getattr(g.message, "usage_metadata", None) or {})

    return _Tap()


def add_guard_args(ap):
    ap.add_argument("--max-usd", type=float, default=10.0,
                    help="hard spend cap for this process; new LLM calls raise past it")
    ap.add_argument("--llm-cache", default=str(Path(__file__).resolve().parent / "llm_cache.sqlite"),
                    help="SQLite LLM cache ('' = off). Re-running a killed command "
                         "replays finished calls for free = resume.")


def enable_guards(args):
    METER.max_usd = args.max_usd
    if args.llm_cache:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache
        set_llm_cache(SQLiteCache(database_path=args.llm_cache))


# ------------------------------------------------------------------------ wiring

def make_chat_anthropic(model, **kwargs):
    """ChatAnthropic factory that drops temperature for Claude-5-family models
    (sonnet-5/opus-5/fable-5), which reject it as deprecated."""
    # flaky network killed 25-min eval runs on one timeout (default retries=2),
    # and a dead socket with no read timeout hung a run for 3h -> bound both
    kwargs.setdefault("max_retries", 6)
    kwargs.setdefault("default_request_timeout", 120)
    kwargs.setdefault("callbacks", [_meter_tap(model)])
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
        ef_search=args.ef_search, sparse_mode=args.sparse_mode)
    grader_llm = make_chat_anthropic(args.grader_model, temperature=0)
    gen_llm = make_chat_anthropic(args.gen_model, temperature=0)
    app = build_crag_app(ensemble, parents, grader_llm, gen_llm,
                         top_k=args.top_k, max_steps=args.max_steps,
                         retry_on_refusal=args.retry_on_refusal,
                         grade_mode=args.grade_mode,
                         grader_chars=args.grader_chars)
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
    ap.add_argument("--sparse-mode", choices=["plain", "or", "bm25"], default="plain",
                    help="KNOB B2: pgvector full-text query. plain = AND of every "
                         "word (matches ~nothing); or = OR of content words; "
                         "bm25 = in-memory BM25 over --chunks (real IDF).")
    ap.add_argument("--grade-mode", choices=["doc", "set"], default="doc",
                    help="KNOB A2: doc = keep/drop each doc; set = one verdict "
                         "on the whole retrieved set, keep all if sufficient.")
    # KNOB A3 (2026-10-07): was a hardcoded 1200 -> grader dropped gold whose
    # answer sat past the cut. 4000 = +18 net correct on 279 positives (v4).
    ap.add_argument("--grader-chars", type=int, default=4000,
                    help="chars of each parent the grader sees (generator sees all)")
    ap.add_argument("--top-k", type=int, default=TOP_K_PARENTS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS,
                    help="refinement retries; 0 = single-shot control (== Stage 5)")
    ap.add_argument("--retry-on-refusal", action="store_true",
                    help="KNOB A1: re-enter the refine loop when the GENERATOR "
                         "refuses in-band (not just when the grader keeps "
                         "nothing). Off = measured baseline.")
    # grader runs on every query over K docs -> the cost/latency hotspot; Haiku
    # is fast + near-frontier and plenty for relevance classification. Generation
    # wants quality -> Sonnet.
    ap.add_argument("--grader-model", default="claude-haiku-4-5")
    ap.add_argument("--gen-model", default="claude-sonnet-5")
    add_guard_args(ap)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ask", help="one-shot question")
    args = ap.parse_args()
    enable_guards(args)

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
