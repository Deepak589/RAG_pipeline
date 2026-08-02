"""Shared retrieval eval harness (Layer A), stage-agnostic. Framework-free:
numpy + stdlib. Lives at RAG/eval/ (not inside any one stage) because it
judges every stage against the same versioned golden set — it is the judge,
not the thing being judged.

Judges retrieval by NUMBERS, not spot-checks. Scores any retriever that can
turn a query into a ranked list of parent_ids. Ships with a parent-child
ranker; add a new `rank_*` function to score another retriever over the same
golden set.

Metrics (per query, then averaged):
    Recall@k  — fraction of relevant parents found in the top-k
    Hit@k     — 1 if any relevant parent is in the top-k, else 0
    MRR       — 1 / rank of the first relevant parent (0 if never found)
Negatives (answer not in corpus) are scored separately: we report the top-1
similarity, so you can later set a refusal threshold that separates them from
real hits.

Run (from RAG/eval/):
    python run_eval.py                 # score stage-2 parent-child retriever
    python run_eval.py --stage3        # score stage-3 (resized) parent-child retriever
    python run_eval.py --selftest      # validate labels + unit-test metrics (no model load)
    python run_eval.py --k 1 3 5 10    # custom cutoffs
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAG_ROOT = HERE.parent                     # RAG/ — holds rag_stage_2/, rag_stage_3/, ...
STAGE_DIR = RAG_ROOT / "rag_stage_2"       # holds parent_child_rag.py
sys.path.insert(0, str(STAGE_DIR))         # import sibling retriever module

QA_PATH = HERE / "qa.json"
RESULTS_DIR = STAGE_DIR / "eval" / "results"
DEFAULT_KS = (1, 3, 5)
CEILING_KS = (20, 50)   # always reported — the recall ceiling is never optional


# --------------------------------------------------------------------- metrics

def recall_at_k(ranked_ids, relevant, k):
    """Fraction of relevant parents present in the top-k."""
    if not relevant:
        return None                       # undefined for negatives
    topk = set(ranked_ids[:k])
    return len(topk & set(relevant)) / len(relevant)


def hit_at_k(ranked_ids, relevant, k):
    """1 if any relevant parent is in the top-k, else 0."""
    if not relevant:
        return None
    return 1.0 if set(ranked_ids[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranked_ids, relevant):
    """1 / rank of the first relevant parent (0 if none in the ranked list)."""
    if not relevant:
        return None
    rel = set(relevant)
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


# ------------------------------------------------------------------- retrievers

def rank_parent_child(index, query):
    """Rank ALL parents for a query, best first.

    Scores every child, collapses to parents by MAX child score, then sorts.
    This is the corrected collapse (no top-8-children cap), so the harness
    measures the retriever's true ceiling rather than a truncated view.

    Returns (ranked_parent_ids, ranked_scores) — full ranking, aligned.
    """
    qvec = index.model.encode([query], normalize_embeddings=True)[0]
    sims = index.matrix @ qvec

    best = {}
    for child, score in zip(index.children, sims):
        pid = child["parent_id"]
        s = float(score)
        if pid not in best or s > best[pid]:
            best[pid] = s

    order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    ids = [pid for pid, _ in order]
    scores = [s for _, s in order]
    return ids, scores


def build_parent_child_ranker():
    """Load the parent-child index and return (name, rank_fn)."""
    import parent_child_rag as pc
    children, parents = pc.load_children(), pc.load_parents()
    index = pc.load_index(children, parents)
    print(f"parent-child index: {len(children)} children -> {len(parents)} parents\n")
    return "parent_child", lambda q: rank_parent_child(index, q)


STAGE3_DIR = STAGE_DIR.parent / "rag_stage_3"
STAGE3_RESULTS_DIR = STAGE3_DIR / "eval" / "results"


def load_stage3_parents():
    """parent_id -> record from stage_3/sections.json (resized parents)."""
    secs = json.loads((STAGE3_DIR / "sections.json").read_text())
    return {f"{s['source']}#{s['section_idx']}": s for s in secs}


def build_parent_child_ranker_stage3():
    """Load a ParentChildIndex over the RESIZED stage-3 files."""
    import parent_child_rag as pc
    children = json.loads((STAGE3_DIR / "chunks.json").read_text())
    parents = load_stage3_parents()
    index = pc.load_index(children, parents, cache_path=STAGE3_DIR / ".dense_cache_pc.npz")
    print(f"stage-3 parent-child index: {len(children)} children -> {len(parents)} parents\n")
    return "parent_child_s3", lambda q: rank_parent_child(index, q)


# -------------------------------------------------------------------- evaluate

def gold_ids(q, stage3=False):
    """Relevant parent_ids for a question. Under --stage3, a question may carry
    a `relevant_parent_ids_s3` override (used where the resized chunker moved
    an answer to a different parent id); otherwise the shared ids apply. This
    keeps one qa.json valid for both the stage-2 baseline and stage-3."""
    if stage3 and "relevant_parent_ids_s3" in q:
        return q["relevant_parent_ids_s3"]
    return q["relevant_parent_ids"]


def evaluate(questions, rank_fn, ks, stage3=False):
    """Run every question through rank_fn, collect per-query metric rows."""
    rows = []
    for q in questions:
        ranked_ids, scores = rank_fn(q["query"])
        relevant = gold_ids(q, stage3)
        row = {
            "id": q["id"], "type": q["type"],
            "top1_score": scores[0] if scores else 0.0,
            "mrr": reciprocal_rank(ranked_ids, relevant),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k)
            row[f"hit@{k}"] = hit_at_k(ranked_ids, relevant, k)
        rows.append(row)
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(rows, ks, types=("factual", "paraphrase")):
    """Average metrics overall and per positive type."""
    metric_keys = ["mrr"] + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks]
    groups = {"ALL_POSITIVE": [r for r in rows if r["type"] in types]}
    for t in types:
        groups[t] = [r for r in rows if r["type"] == t]

    summary = {}
    for name, grp in groups.items():
        if not grp:
            continue
        summary[name] = {"n": len(grp)}
        for mk in metric_keys:
            summary[name][mk] = _mean([r[mk] for r in grp])
    return summary, metric_keys


# ---------------------------------------------------------------------- report

def print_report(rows, summary, metric_keys, ks):
    cols = ["mrr"] + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks]
    w = max(len(c) for c in cols) + 2

    print("=" * 72)
    print("RETRIEVAL EVAL — Layer A (retrieval only, deterministic)")
    print("=" * 72)
    header = f"{'group':<16}{'n':>4}  " + "".join(f"{c:>{w}}" for c in cols)
    print(header)
    print("-" * len(header))
    for name in ["ALL_POSITIVE", "factual", "paraphrase"]:
        if name not in summary:
            continue
        s = summary[name]
        line = f"{name:<16}{s['n']:>4}  " + "".join(f"{s[c]:>{w}.3f}" for c in cols)
        print(line)

    # Negatives: report score separation so a refusal threshold can be chosen.
    negs = [r for r in rows if r["type"] == "negative"]
    poss = [r for r in rows if r["type"] in ("factual", "paraphrase")]
    if negs:
        print("-" * len(header))
        neg_top1 = _mean([r["top1_score"] for r in negs])
        pos_top1 = _mean([r["top1_score"] for r in poss])
        print(f"negatives (n={len(negs)}): mean top-1 score = {neg_top1:.3f}   "
              f"| positives mean top-1 = {pos_top1:.3f}")
        print(f"  separation = {pos_top1 - neg_top1:+.3f}  "
              f"(bigger gap = easier to threshold refusals)")
    print("=" * 72)


def save_results(name, rows, summary, ks, stage3=False):
    results_dir = STAGE3_RESULTS_DIR if stage3 else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = results_dir / f"{name}_{stamp}.json"
    out.write_text(json.dumps(
        {"retriever": name, "ks": list(ks), "summary": summary, "rows": rows},
        indent=2, ensure_ascii=False,
    ))
    print(f"\nsaved -> {out}")


# ---------------------------------------------------------------------- selftest

def validate_labels(questions, stage3=False):
    """Every relevant_parent_id must exist in the parent store."""
    if stage3:
        parents = load_stage3_parents()
    else:
        import parent_child_rag as pc
        parents = pc.load_parents()
    bad = []
    for q in questions:
        for pid in gold_ids(q, stage3):
            if pid not in parents:
                bad.append((q["id"], pid))
    if bad:
        print("LABEL ERRORS — these parent_ids do not exist in sections.json:")
        for qid, pid in bad:
            print(f"  {qid}: {pid}")
        return False
    print(f"labels OK — all {sum(len(q['relevant_parent_ids']) for q in questions)} "
          f"parent_ids across {len(questions)} questions resolve to real sections")
    return True


def selftest_metrics():
    """Unit-test the metric functions on hand-checked rankings."""
    ranked = ["a", "b", "c", "d"]
    checks = [
        (recall_at_k(ranked, ["b"], 3), 1.0, "recall: single relevant in top-3"),
        (recall_at_k(ranked, ["b", "d"], 3), 0.5, "recall: 1 of 2 in top-3"),
        (recall_at_k(ranked, ["x"], 4), 0.0, "recall: relevant absent"),
        (hit_at_k(ranked, ["c"], 3), 1.0, "hit: relevant at rank 3"),
        (hit_at_k(ranked, ["d"], 3), 0.0, "hit: relevant at rank 4, k=3"),
        (reciprocal_rank(ranked, ["a"]), 1.0, "mrr: first pos"),
        (reciprocal_rank(ranked, ["c"]), 1 / 3, "mrr: rank 3"),
        (reciprocal_rank(ranked, ["x"]), 0.0, "mrr: never found"),
        (recall_at_k(ranked, [], 3), None, "negative: recall undefined"),
    ]
    ok = True
    for got, want, label in checks:
        good = got is None and want is None or (
            got is not None and want is not None and abs(got - want) < 1e-9)
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: got {got}")
        ok = ok and good
    return ok


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_KS),
                    help=f"cutoffs to report; ceiling {CEILING_KS} always added")
    ap.add_argument("--retriever", choices=["parent_child", "reranked"],
                    default="parent_child", help="which retriever to score")
    ap.add_argument("--rerank-depth", type=int, default=20,
                    help="dense children reranked per query (reranked only)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate labels + unit-test metrics (no model load)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--stage3", action="store_true",
                    help="score the resized stage-3 chunker instead of stage-2")
    args = ap.parse_args()
    # Ceiling cutoffs are always included so a reranker that truncates the
    # candidate pool can't hide a dropped recall ceiling behind a good hit@3.
    ks = tuple(sorted(set(args.k) | set(CEILING_KS)))

    data = json.loads(QA_PATH.read_text())
    questions = data["questions"]

    if args.selftest:
        print("== metric unit tests ==")
        m_ok = selftest_metrics()
        print("\n== label validation ==")
        l_ok = validate_labels(questions, stage3=args.stage3)
        sys.exit(0 if (m_ok and l_ok) else 1)

    if args.retriever == "reranked":
        sys.path.insert(0, str(RAG_ROOT))
        from reranker import build_reranked_ranker
        name, rank_fn = build_reranked_ranker(args.rerank_depth, stage3=args.stage3)
    elif args.stage3:
        name, rank_fn = build_parent_child_ranker_stage3()
    else:
        name, rank_fn = build_parent_child_ranker()
    rows = evaluate(questions, rank_fn, ks, stage3=args.stage3)
    summary, metric_keys = aggregate(rows, ks)
    print_report(rows, summary, metric_keys, ks)
    if not args.no_save:
        save_results(name, rows, summary, ks, stage3=args.stage3)


if __name__ == "__main__":
    main()
