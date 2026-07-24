"""Stage 2 — parent-child dense RAG: retrieve small children, feed parents.

Pipeline (all built by earlier stages):
    pdf_extractor.py -> sections.json   (parents: full sections)
    chunker.py       -> chunks.json     (children: 120-word windows)
    this file        -> embed children, retrieve, hand parents to the LLM

Why parent-child: a 120-word child embeds to a sharp, specific vector, so
retrieval lands on the right spot. But a lone window loses context, so we
return the child's *parent section* to the generator. Precision of small
chunks, context of large ones.

Ollama generation is reused from Naive_rag.py (identical local call);
retrieval + prompting here are parent-child specific.

Run:  python parent_child_rag.py --query "what is dense passage retrieval?"
      python parent_child_rag.py                 # interactive
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))   # reach shared generator.py at RAG/ root
from generator import EMBED_MODEL, build_prompt as _prompt, generate

HERE = Path(__file__).parent
CHUNKS_PATH = HERE / "chunks.json"       # children
SECTIONS_PATH = HERE / "sections.json"   # parents
CACHE_PATH = HERE / ".dense_cache_pc.npz"

TOP_CHILDREN = 8     # children scored, then collapsed to their parents
TOP_PARENTS = 3      # unique parent sections handed to the generator


# --------------------------------------------------------------- load parents

def load_parents():
    """Map parent_id -> parent section record (from sections.json)."""
    sections = json.loads(SECTIONS_PATH.read_text())
    return {f"{s['source']}#{s['section_idx']}": s for s in sections}


def load_children():
    """Child chunk records from chunks.json."""
    return json.loads(CHUNKS_PATH.read_text())


# ------------------------------------------------------------ parent-child index

class ParentChildIndex:
    """Dense retrieval over child chunks that returns whole parent sections."""

    def __init__(self, children, parents, matrix=None, model_name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.children = children
        self.parents = parents
        self.model = SentenceTransformer(model_name)
        if matrix is None:
            texts = [c["text"] for c in children]
            matrix = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        self.matrix = np.asarray(matrix)

    def retrieve(self, query):
        """Top child hits collapsed to their unique parent sections.

        Returns list of (score, parent_record) best first, where score is
        the best child score for that parent.
        """
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.matrix @ qvec
        order = np.argsort(sims)[::-1][:TOP_CHILDREN]

        seen, results = set(), []
        for i in order:
            pid = self.children[i]["parent_id"]
            if pid in seen:
                continue
            seen.add(pid)
            results.append((float(sims[i]), self.parents[pid]))
            if len(results) == TOP_PARENTS:
                break
        return results


def _fingerprint(children, model_name):
    """Hash of child texts + model — cache is stale if either changes."""
    h = hashlib.sha256(model_name.encode())
    for c in children:
        h.update(c["text"].encode())
        h.update(b"\0")
    return h.hexdigest()


def load_index(children, parents, cache_path=CACHE_PATH):
    """Build a ParentChildIndex, reusing cached child vectors when valid."""
    fp = _fingerprint(children, EMBED_MODEL)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if str(data["fingerprint"]) == fp:
            print(f"Loaded cached vectors from {cache_path.name}")
            return ParentChildIndex(children, parents, matrix=data["matrix"])
    index = ParentChildIndex(children, parents)
    np.savez(cache_path, matrix=index.matrix, fingerprint=fp)
    print(f"Computed + cached vectors to {cache_path.name}")
    return index


# ------------------------------------------------------------------ generation

def build_prompt(question, parents):
    """Feed parent sections into the shared grounded prompt."""
    # _prompt expects (score, source, text) triples.
    retrieved = [(score, p["source"], p["text"]) for score, p in parents]
    return _prompt(question, retrieved)


def answer(index, question):
    parents = index.retrieve(question)
    print("\nRetrieved parents (via best child):")
    for score, p in parents:
        print(f"  {score:.3f}  {p['source'][:32]}  §\"{p['title'][:38]}\"  "
              f"p{p['page_start']}-{p['page_end']}")

    result, reason = generate(build_prompt(question, parents))
    if result is None:
        print(f"\n[{reason} — showing retrieved context only]")
        for _, p in parents:
            print(f"\n--- {p['source']} §{p['title']} ---\n{p['text'][:800]}...")
    else:
        print(f"\nAnswer: {result}")


# ------------------------------------------------------------------------ main

def main():
    if not CHUNKS_PATH.exists() or not SECTIONS_PATH.exists():
        sys.exit("Missing chunks.json / sections.json — run pdf_extractor.py "
                 "then chunker.py first")

    children, parents = load_children(), load_parents()
    index = load_index(children, parents)
    print(f"Parent-child index ({EMBED_MODEL}): "
          f"{len(children)} children -> {len(parents)} parents")

    args = sys.argv[1:]
    if len(args) > 1 and args[0] == "--query":
        answer(index, args[1])
        return

    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            answer(index, q)


if __name__ == "__main__":
    main()
