"""Stage 4 — ask the LLM using the CURRENT serving retriever.

Retrieval = stage-4 hybrid (BM25 + dense, RRF) over BLURBED children — the
locked stage-4 serving pipeline (R@5 0.833, R@1 0.531, MRR 0.672). The blurb is
an INDEX-TIME trick: it sharpens retrieval, but the text handed to the generator
is the ORIGINAL parent section, unchanged. The LLM never sees a blurb.

Pipeline:
    query -> hybrid rank_fn -> top-3 parent_ids -> parent-section text
          -> grounded prompt -> local Ollama (qwen3.5) -> answer

Retrieval and generation are decoupled: retrieval needs sentence-transformers
(dense) + numpy (BM25); generation needs Ollama. If Ollama is down/slow the
retrieved parent sections are printed instead of crashing — same graceful-
degrade contract as Naive_rag.py / parent_child_rag.py.

Run (sentence-transformers installed, Ollama up):
    python answer.py --query "what is dense passage retrieval?"
    python answer.py                          # interactive REPL
    python answer.py --query "..." --show-blurb   # also show the blurb that indexed each hit
    python answer.py --query "..." --raw-bm25     # ablation: BM25 on raw text, dense on blurbs
    python answer.py --query "..." --rrf-k 40 --top 5
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))          # RAG/ root -> shared generator.py
import hybrid                                  # stage-4 hybrid.py (this same dir)
from generator import build_prompt, generate

SECTIONS_PATH = HERE / "sections.json"
CHUNKS_PATH = HERE / "chunks.json"
DEFAULT_TOP_PARENTS = 3


def load_parents():
    """parent_id -> section record (the text actually served to the LLM)."""
    secs = json.loads(SECTIONS_PATH.read_text())
    return {f"{s['source']}#{s['section_idx']}": s for s in secs}


def best_blurb_per_parent():
    """parent_id -> the blurb of its first child (display only, not served)."""
    out = {}
    for c in json.loads(CHUNKS_PATH.read_text()):
        out.setdefault(c["parent_id"], c.get("blurb", ""))
    return out


def answer(rank_fn, parents, question, top_parents, show_blurb=False, blurbs=None):
    ranked_ids, scores = rank_fn(question)
    top = list(zip(ranked_ids, scores))[:top_parents]

    print("\nRetrieved parents (hybrid over blurbed children):")
    retrieved = []
    for pid, score in top:
        p = parents[pid]
        print(f"  {score:.4f}  {p['source'][:30]:<30}  "
              f"§\"{p.get('title', '')[:38]}\"  "
              f"p{p.get('page_start', '?')}-{p.get('page_end', '?')}")
        if show_blurb and blurbs:
            print(f"            blurb: {blurbs.get(pid, '')}")
        # Serve the ORIGINAL parent section text, NOT the blurbed child text.
        retrieved.append((score, p["source"], p["text"]))

    result, reason = generate(build_prompt(question, retrieved))
    if result is None:
        print(f"\n[{reason} — showing retrieved context only]")
        for _, src, text in retrieved:
            print(f"\n--- {src} ---\n{text[:800]}...")
    else:
        print(f"\nAnswer: {result}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="one-shot question (omit for interactive REPL)")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP_PARENTS,
                    help="parent sections passed to the LLM (default 3)")
    ap.add_argument("--rrf-k", type=int, default=hybrid.RRF_K,
                    help="RRF damping constant (default 60)")
    ap.add_argument("--raw-bm25", dest="blurb_bm25", action="store_false", default=True,
                    help="BM25 indexes raw text; only dense sees blurbs (ablation)")
    ap.add_argument("--show-blurb", action="store_true",
                    help="print the index-time blurb of each retrieved parent")
    args = ap.parse_args()

    if not CHUNKS_PATH.exists():
        sys.exit("chunks.json missing — run: python blurb_gen.py")

    _name, rank_fn = hybrid.build_hybrid_ranker(blurb_bm25=args.blurb_bm25, k=args.rrf_k)
    parents = load_parents()
    blurbs = best_blurb_per_parent() if args.show_blurb else None

    if args.query:
        answer(rank_fn, parents, args.query, args.top, args.show_blurb, blurbs)
        return

    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            answer(rank_fn, parents, q, args.top, args.show_blurb, blurbs)


if __name__ == "__main__":
    main()
