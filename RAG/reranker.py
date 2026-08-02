"""Shared cross-encoder reranker, stage-agnostic — lives at RAG/ root (not
inside any one stage) because it reranks over EITHER stage's dense-retrieved
children, picked by the `stage3` flag.

Dense retrieval lands on the right parent's chunk but ranks it low on this
homogeneous corpus (all 7 papers concern RAG / retrieval / attention, so parent
sections are near-duplicates in embedding space). A cross-encoder rescores the
top-N dense children by jointly encoding (query, chunk) — far better at
discriminating the right passage — then we collapse to parents by best reranked
score.

Eval-only: registered as the "reranked" retriever in eval/run_eval.py.
parent_child_rag.py runtime behavior is untouched.

Run (via the harness, from RAG/eval/):
    python run_eval.py --retriever reranked
    python run_eval.py --retriever reranked --rerank-depth 30
    python run_eval.py --retriever reranked --stage3
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "rag_stage_2"))  # reach parent_child_rag
import parent_child_rag as pc

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_DEPTH = 20     # swept 1..50: ~20 is best; 50 adds noise that hurts hit@1


# ----------------------------------------------------------------------- ranker

def rank_reranked(index, cross_encoder, query, depth=DEFAULT_DEPTH):
    """Rerank the top-`depth` dense children, then keep the dense tail below.

    Two parts, concatenated (not score-merged — CE scores and cosine are on
    different scales, so we join by *rank*, not value):
      HEAD  parents seen in the top-`depth` children, ordered by best CE score.
      TAIL  every remaining parent, in original dense order (best child cosine).

    Reranking only reorders the head — the head is all the LLM's top-3 will ever
    draw from. The tail exists so a parent whose best child fell outside `depth`
    is never *evicted* (which would silently cap recall@20/@50 below the dense
    ceiling). Appending it keeps the ceiling intact at zero quality cost.

    Returns (ranked_parent_ids, ranked_scores) best first — same contract as
    parent_child_rag's rank helper. Note: scores[0] is the head's top CE score;
    scores are not comparable across the head/tail boundary (ranking is).
    """
    qvec = index.model.encode([query], normalize_embeddings=True)[0]
    sims = index.matrix @ qvec

    # HEAD — cross-encode the top-`depth` children, collapse to parents by max CE.
    top = np.argsort(sims)[::-1][:depth]
    ce_scores = cross_encoder.predict(
        [(query, index.children[i]["text"]) for i in top]
    )
    head_best = {}
    for i, score in zip(top, ce_scores):
        pid = index.children[i]["parent_id"]
        s = float(score)
        if pid not in head_best or s > head_best[pid]:
            head_best[pid] = s
    head = sorted(head_best.items(), key=lambda kv: kv[1], reverse=True)

    # TAIL — full dense parent order (all children collapsed by best cosine),
    # minus parents already in the head, in dense rank order.
    dense_best = {}
    for child, score in zip(index.children, sims):
        pid = child["parent_id"]
        s = float(score)
        if pid not in dense_best or s > dense_best[pid]:
            dense_best[pid] = s
    head_ids = {pid for pid, _ in head}
    tail = [(pid, s) for pid, s in
            sorted(dense_best.items(), key=lambda kv: kv[1], reverse=True)
            if pid not in head_ids]

    ranked = head + tail
    ids = [pid for pid, _ in ranked]
    scores = [s for _, s in ranked]
    return ids, scores


def build_reranked_ranker(depth=DEFAULT_DEPTH, stage3=False):
    """Load dense index + cross-encoder; return (name, rank_fn) for the harness.

    stage3=True reranks over the resized stage-3 chunks/sections instead of
    the stage-2 baseline, so reranking can be A/B'd on top of the chunk resize.
    """
    from sentence_transformers import CrossEncoder

    if stage3:
        stage3_dir = Path(__file__).parent / "rag_stage_3"
        children = json.loads((stage3_dir / "chunks.json").read_text())
        secs = json.loads((stage3_dir / "sections.json").read_text())
        parents = {f"{s['source']}#{s['section_idx']}": s for s in secs}
        index = pc.load_index(children, parents, cache_path=stage3_dir / ".dense_cache_pc.npz")
        name = "reranked_s3"
    else:
        children, parents = pc.load_children(), pc.load_parents()
        index = pc.load_index(children, parents)
        name = "reranked"
    ce = CrossEncoder(RERANK_MODEL)
    print(f"reranker: {RERANK_MODEL}, depth={depth}, "
          f"{len(children)} children -> {len(parents)} parents\n")
    return name, lambda q: rank_reranked(index, ce, q, depth)
