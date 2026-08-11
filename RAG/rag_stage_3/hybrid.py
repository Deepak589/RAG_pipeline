"""Stage 3 — hybrid retrieval: BM25 (lexical) + dense, fused by RRF.

Why hybrid: on this homogeneous corpus the worst dense misses are exact-term
queries (names, IDs, rare tokens — "DPR", "BART", "BM25") that a bi-encoder
blurs but lexical matching nails. BM25 covers dense's blind spot; dense covers
BM25's (paraphrase / synonym). Fusing the two should lift the recall ceiling
above either alone.

Fusion is RANK-based (Reciprocal Rank Fusion), not score-based: BM25 scores and
cosine similarities live on different, incomparable scales, so we combine the
two *rankings* instead of the two score vectors.

    RRF(parent) = 1/(k + rank_dense) + 1/(k + rank_bm25)        (k = 60)

Both retrievers rank the FULL parent set, so every parent has a rank in each
list — no missing-member special case. Retrieve wide here, rerank later: this
module is the retrieval stage; the cross-encoder is a separate refinement stage
that reorders this pool (see reranker.py). They compose in sequence.

Contract matches the harness: build_hybrid_ranker() -> (name, rank_fn) where
rank_fn(query) -> (ranked_parent_ids, ranked_scores), full ranking, best first.

Run (via the harness):
    python ../eval/run_eval.py --retriever hybrid --stage3
    python ../eval/run_eval.py --retriever hybrid --stage3 --rrf-k 60
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_stage_2"))
import parent_child_rag as pc

RRF_K = 60          # RRF damping; larger k flattens the weight of top ranks
BM25_K1 = 1.5       # term-frequency saturation
BM25_B = 0.75       # length normalisation strength

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alphanumeric tokens. Keeps digits and short IDs ('dpr',
    'bm25', '2014') — exactly the exact-match terms dense blurs."""
    return _TOKEN_RE.findall(text.lower())


# ----------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25 over child chunks. Pure numpy/stdlib — no model, no network."""

    def __init__(self, children, k1=BM25_K1, b=BM25_B):
        self.children = children
        self.k1, self.b = k1, b
        self.docs = [tokenize(c["text"]) for c in children]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = self.doc_len.mean() if len(self.docs) else 0.0

        # document frequency per term
        df = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        n = len(self.docs)
        # BM25+ idf: always positive, avoids negative weights for common terms
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

        # per-doc term frequencies
        self.tf = []
        for doc in self.docs:
            counts = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def scores(self, query):
        """BM25 score of every child for `query`. Returns np.array[len(children)]."""
        q_terms = [t for t in tokenize(query) if t in self.idf]
        out = np.zeros(len(self.children), dtype=float)
        if not q_terms:
            return out
        for i, counts in enumerate(self.tf):
            denom_len = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            s = 0.0
            for t in q_terms:
                f = counts.get(t, 0)
                if f:
                    s += self.idf[t] * (f * (self.k1 + 1)) / (f + denom_len)
            out[i] = s
        return out


# ------------------------------------------------------- collapse + rank helpers

def _parents_by_max_child(children, per_child_score):
    """Collapse child scores to parents by MAX, return parent_ids ranked best
    first (stable sort, so ties are deterministic)."""
    best = {}
    for child, s in zip(children, per_child_score):
        pid = child["parent_id"]
        s = float(s)
        if pid not in best or s > best[pid]:
            best[pid] = s
    order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in order]


def rank_hybrid(index, bm25, query, k=RRF_K):
    """Dense ranking + BM25 ranking, fused by RRF over parents.

    Returns (ranked_parent_ids, ranked_scores) — full ranking, best first.
    Score is the RRF value (not a similarity); comparable within a query only.
    """
    # dense: parents by best child cosine
    qvec = index.model.encode([query], normalize_embeddings=True)[0]
    sims = index.matrix @ qvec
    dense_order = _parents_by_max_child(index.children, sims)

    # lexical: parents by best child BM25
    bm25_order = _parents_by_max_child(index.children, bm25.scores(query))

    dense_rank = {pid: r for r, pid in enumerate(dense_order, start=1)}
    bm25_rank = {pid: r for r, pid in enumerate(bm25_order, start=1)}

    fused = {}
    for pid in set(dense_rank) | set(bm25_rank):
        rd = dense_rank.get(pid, len(dense_rank) + 1)
        rb = bm25_rank.get(pid, len(bm25_rank) + 1)
        fused[pid] = 1.0 / (k + rd) + 1.0 / (k + rb)

    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in order], [s for _, s in order]


# --------------------------------------------------------------------- builders

def build_hybrid_ranker(stage3=False, k=RRF_K):
    """Load dense index + build BM25 over the same children; return
    (name, rank_fn) for the harness. stage3=True uses the resized stage-3
    chunks/sections so hybrid is measured on the same corpus as dense-s3."""
    here = Path(__file__).parent
    if stage3:
        children = json.loads((here / "chunks.json").read_text())
        secs = json.loads((here / "sections.json").read_text())
        parents = {f"{s['source']}#{s['section_idx']}": s for s in secs}
        index = pc.load_index(children, parents, cache_path=here / ".dense_cache_pc.npz")
        name = "hybrid_s3"
    else:
        children, parents = pc.load_children(), pc.load_parents()
        index = pc.load_index(children, parents)
        name = "hybrid"
    bm25 = BM25(children)
    print(f"hybrid: dense + BM25 (RRF k={k}), "
          f"{len(children)} children -> {len(parents)} parents\n")
    return name, lambda q: rank_hybrid(index, bm25, q, k)


def build_bm25_ranker(stage3=False):
    """BM25-ONLY ranker (no model) — for isolating the lexical contribution and
    for measuring in environments without the embedding model. Same contract."""
    here = Path(__file__).parent
    if stage3:
        children = json.loads((here / "chunks.json").read_text())
    else:
        children = pc.load_children()
    bm25 = BM25(children)
    print(f"bm25-only: {len(children)} children\n")

    def rank_fn(query):
        scores = bm25.scores(query)
        best = {}
        for child, s in zip(children, scores):
            pid = child["parent_id"]
            s = float(s)
            if pid not in best or s > best[pid]:
                best[pid] = s
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [pid for pid, _ in order], [s for _, s in order]

    return ("bm25_s3" if stage3 else "bm25"), rank_fn
