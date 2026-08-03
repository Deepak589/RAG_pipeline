"""Stage 4 — hybrid retrieval over BLURBED children.

Identical fusion logic to stage 3 (BM25 + dense, RRF) — the ONLY change under
test is that children now carry a prepended context blurb (see blurb_gen.py).
We reuse stage-3's BM25 class and rank_hybrid verbatim; nothing about the
retriever math changes, so any metric move is attributable to the blurb alone.

The one honest knob to measure: does the blurb belong in the BM25 text too, or
only in the dense embedding? Blurbs add real keywords (helps thin chunks) but
repeat the section's terms across every child (inflates doc_len, shifts idf).
    --blurb-bm25 on  (default): BM25 tokenizes the blurbed text (child["text"])
    --blurb-bm25 off          : BM25 tokenizes raw_text; only dense sees blurbs
Dense ALWAYS embeds the blurbed text (that is the stage-4 hypothesis).

Contract matches the harness: build_*_ranker(...) -> (name, rank_fn) where
rank_fn(query) -> (ranked_parent_ids, ranked_scores), full ranking, best first.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_stage_2"))
import parent_child_rag as pc


def _load_stage3_hybrid():
    """Import stage-3 hybrid.py by explicit path under a unique module name,
    so it never collides with this file (both are named hybrid.py)."""
    path = Path(__file__).parent.parent / "rag_stage_3" / "hybrid.py"
    spec = importlib.util.spec_from_file_location("hybrid_s3_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s3 = _load_stage3_hybrid()
BM25, RRF_K, rank_hybrid = _s3.BM25, _s3.RRF_K, _s3.rank_hybrid   # reused verbatim

HERE = Path(__file__).parent
CHUNKS_PATH = HERE / "chunks.json"
SECTIONS_PATH = HERE / "sections.json"
CACHE_PATH = HERE / ".dense_cache_pc.npz"


def _load():
    children = json.loads(CHUNKS_PATH.read_text())
    secs = json.loads(SECTIONS_PATH.read_text())
    parents = {f"{s['source']}#{s['section_idx']}": s for s in secs}
    return children, parents


def _raw_view(children):
    """Children with text := raw_text, so BM25 indexes the un-blurbed window."""
    out = []
    for c in children:
        d = dict(c)
        d["text"] = c.get("raw_text", c["text"])
        out.append(d)
    return out


def build_hybrid_ranker(blurb_bm25=True, k=RRF_K):
    """Stage-4 hybrid: dense over blurbed text + BM25 over blurbed-or-raw text."""
    children, parents = _load()
    # dense: fingerprint hashes child["text"] (blurbed) -> cache auto-invalidates
    index = pc.load_index(children, parents, cache_path=CACHE_PATH)
    bm25_children = children if blurb_bm25 else _raw_view(children)
    bm25 = BM25(bm25_children)
    tag = "blurbBM25" if blurb_bm25 else "rawBM25"
    name = f"hybrid_s4_{tag}"
    print(f"{name}: dense(blurbed) + BM25({'blurbed' if blurb_bm25 else 'raw'}) "
          f"RRF k={k}, {len(children)} children -> {len(parents)} parents\n")
    return name, lambda q: rank_hybrid(index, bm25, q, k)


def build_bm25_ranker(blurb_bm25=True):
    """BM25-only over stage-4 children (isolates the lexical side of the blurb)."""
    children, parents = _load()
    bm25_children = children if blurb_bm25 else _raw_view(children)
    bm25 = BM25(bm25_children)
    tag = "blurbBM25" if blurb_bm25 else "rawBM25"
    name = f"bm25_s4_{tag}"
    print(f"{name}: {len(children)} children\n")

    def rank_fn(query):
        scores = bm25.scores(query)
        best = {}
        for child, s in zip(bm25_children, scores):
            pid = child["parent_id"]
            s = float(s)
            if pid not in best or s > best[pid]:
                best[pid] = s
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [pid for pid, _ in order], [s for _, s in order]

    return name, rank_fn
