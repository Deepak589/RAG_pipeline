"""Diagnose why the cross-encoder demotes specific questions on stage-3.

For each target question it prints, side by side:
  DENSE   — parent ranking by best child cosine, with the gold parent's rank
            and its best child flagged.
  HEAD    — the top-`depth` dense children fed to the cross-encoder, each with
            its CE score, dense rank, parent_id, and whether it is the gold
            parent's child. This is where a demotion is visible: read off the
            gold child's CE score vs whatever the CE put at head rank 1.
  FINAL   — the collapsed parent ranking after rerank, gold parent's rank.

Answers the question "granularity vs synonym punishment" directly:
  - gold child short + low CE + paraphrase query  -> granularity / lexical miss
  - gold child long, high dense rank, CE promotes a near-duplicate -> CE
    over-confidence on a homogeneous corpus.

Run from RAG/rag_stage_3/ (same env that runs the eval):
    python diagnose_demotions.py
    python diagnose_demotions.py --depth 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_stage_2"))
import parent_child_rag as pc

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# (id, query, gold_parent_id) — the three stage-3 rerank demotions.
TARGETS = [
    ("lewis-models",
     "What is the difference between RAG-Sequence and RAG-Token models?",
     "Lewis et al. - 2020 - Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks.pdf#3"),
    ("vas-optim-para",
     "What method ramps the training step size up early then decays it as training progresses?",
     "Vaswani et al. - 2017 - Attention Is All You Need.pdf#14"),
    ("realm-train-para",
     "How does REALM cope with summing over every document in the corpus being intractable while learning?",
     "Guu et al. - 2020 - REALM - Retrieval-Augmented Language Model Pre-Training.pdf#6"),
]


def load_index():
    here = Path(__file__).parent
    children = json.loads((here / "chunks.json").read_text())
    secs = json.loads((here / "sections.json").read_text())
    parents = {f"{s['source']}#{s['section_idx']}": s for s in secs}
    index = pc.load_index(children, parents, cache_path=here / ".dense_cache_pc.npz")
    return index


def short(pid, n=48):
    """Trim the long PDF filename to '...key#idx' for readable columns."""
    base, _, tail = pid.rpartition("#")
    stem = base.split(" - ")[0][:n]
    return f"{stem}…#{tail}"


def diagnose(index, ce, query, gold_pid, depth):
    qvec = index.model.encode([query], normalize_embeddings=True)[0]
    sims = index.matrix @ qvec
    dense_rank = np.argsort(sims)[::-1]

    # DENSE parent ranking (max child cosine)
    dense_best = {}
    for child, s in zip(index.children, sims):
        p = child["parent_id"]
        if p not in dense_best or s > dense_best[p]:
            dense_best[p] = float(s)
    dense_parents = [p for p, _ in sorted(dense_best.items(), key=lambda kv: kv[1], reverse=True)]
    gold_dense_rank = dense_parents.index(gold_pid) + 1 if gold_pid in dense_parents else -1

    print(f"  DENSE: gold parent rank = {gold_dense_rank}  "
          f"(best child cosine = {dense_best.get(gold_pid, float('nan')):.4f})")

    # HEAD — the CE's working set
    top = dense_rank[:depth]
    ce_scores = ce.predict([(query, index.children[i]["text"]) for i in top])
    print(f"  HEAD (top-{depth} children by CE score):")
    order = sorted(range(len(top)), key=lambda j: ce_scores[j], reverse=True)
    for hp, j in enumerate(order, start=1):
        i = top[j]
        c = index.children[i]
        is_gold = "  <== GOLD" if c["parent_id"] == gold_pid else ""
        print(f"    ce#{hp:<2} ce={ce_scores[j]:+7.3f}  dense#{list(dense_rank).index(i)+1:<3} "
              f"len={len(c['text']):<4} {short(c['parent_id'])}{is_gold}")

    # FINAL — collapse head by max CE, append dense tail
    head_best = {}
    for j, i in enumerate(top):
        p = index.children[i]["parent_id"]
        s = float(ce_scores[j])
        if p not in head_best or s > head_best[p]:
            head_best[p] = s
    head = [p for p, _ in sorted(head_best.items(), key=lambda kv: kv[1], reverse=True)]
    tail = [p for p in dense_parents if p not in set(head)]
    final = head + tail
    gold_final_rank = final.index(gold_pid) + 1 if gold_pid in final else -1
    print(f"  FINAL: gold parent rank = {gold_final_rank}"
          f"   (dense {gold_dense_rank} -> rerank {gold_final_rank})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    index = load_index()
    ce = CrossEncoder(RERANK_MODEL)

    for qid, query, gold in TARGETS:
        print("=" * 96)
        print(f"{qid}\n  Q: {query}\n  gold: {short(gold)}")
        diagnose(index, ce, query, gold, args.depth)
        print()


if __name__ == "__main__":
    main()
