"""Naive RAG from scratch: chunk -> embed -> cosine retrieve -> generate.

No RAG frameworks. numpy + sentence-transformers + stdlib.
Generation uses a local Ollama model; falls back to printing retrieved
chunks if Ollama is not running.
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

from generator import EMBED_MODEL, build_prompt, generate

DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_SIZE = 200      # words per chunk
CHUNK_OVERLAP = 40    # words shared between consecutive chunks
TOP_K = 3
CACHE_PATH = Path(__file__).parent / ".dense_cache.npz"   # persisted chunk vectors


# ---------------------------------------------------------------- load + chunk

def _word_chunks(words):
    """Yield overlapping fixed-size word windows, skipping tiny tails."""
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE]
        if len(piece) < 20:  # skip tiny tail fragments
            continue
        yield " ".join(piece)


def _md_sections(text):
    """Split markdown on '##' headers into (heading, body) sections."""
    sections = []
    heading, body = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if body:
                sections.append((heading, "\n".join(body)))
            heading, body = line[3:].strip(), [line]
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body)))
    return sections


def load_and_chunk():
    """Read every .md in DOCS_DIR and split into chunks.

    Split on '##' headers first, then word-chunk within each section
    (keeps topics intact, avoids mid-idea cuts across sections).
    Each chunk is prefixed with its section heading so later windows
    still carry the topic they belong to.

    Navigation sections (table of contents) are skipped: they list every
    heading in the doc, so they match almost any query and crowd out the
    section that actually answers it.

    Returns a list of (source_filename, chunk_text) tuples.
    """
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for heading, body in _md_sections(path.read_text()):
            if heading and "table of contents" in heading.lower():
                continue
            for piece in _word_chunks(body.split()):
                if heading and not piece.startswith(f"## {heading}"):
                    piece = f"## {heading}\n\n{piece}"
                chunks.append((path.name, piece))
    return chunks


# ------------------------------------------------------------------- embedding

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfIndex:
    """Hand-rolled TF-IDF over chunks.

    TF  = term count / chunk length
    IDF = log((1 + N) / (1 + df)) + 1
    Rows are L2-normalized so cosine similarity is just a dot product.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        docs_tokens = [tokenize(text) for _, text in chunks]

        # vocabulary: every word that appears in any chunk
        self.vocab = {w: i for i, w in enumerate(sorted(set(w for toks in docs_tokens for w in toks)))}

        # document frequency: in how many chunks does each word appear
        n = len(chunks)
        df = np.zeros(len(self.vocab))
        for toks in docs_tokens:
            for w in set(toks):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1

        # chunk matrix: one TF-IDF row per chunk
        self.matrix = np.zeros((n, len(self.vocab)))
        for row, toks in enumerate(docs_tokens):
            for w in toks:
                self.matrix[row, self.vocab[w]] += 1
            self.matrix[row] /= len(toks)          # TF
        self.matrix *= self.idf                    # TF * IDF
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True)

    def embed_query(self, query):
        toks = tokenize(query)
        vec = np.zeros(len(self.vocab))
        for w in toks:
            if w in self.vocab:                    # unseen words carry no signal
                vec[self.vocab[w]] += 1
        if not vec.any():
            return vec
        vec = (vec / len(toks)) * self.idf
        return vec / np.linalg.norm(vec)

    def retrieve(self, query, k=TOP_K):
        """Return top-k (score, source, chunk_text), best first."""
        sims = self.matrix @ self.embed_query(query)
        top = np.argsort(sims)[::-1][:k]
        return [(sims[i], *self.chunks[i]) for i in top]


class DenseIndex:
    """Dense semantic retrieval via sentence-transformers bi-encoder.

    Each chunk is a normalized embedding; cosine similarity is a dot product.
    Handles synonyms/paraphrase that lexical matching misses (RAG_GUIDE.md §5.2).
    """

    def __init__(self, chunks, matrix=None, model_name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)   # needed to embed queries
        if matrix is None:
            texts = [text for _, text in chunks]
            matrix = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        self.matrix = np.asarray(matrix)

    def retrieve(self, query, k=TOP_K):
        """Return top-k (score, source, chunk_text), best first."""
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.matrix @ qvec
        top = np.argsort(sims)[::-1][:k]
        return [(sims[i], *self.chunks[i]) for i in top]


def _fingerprint(chunks, model_name):
    """Hash of chunk texts + model — cache is stale if either changes."""
    h = hashlib.sha256(model_name.encode())
    for _src, text in chunks:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


def load_dense_index(chunks, cache_path=CACHE_PATH):
    """Build a DenseIndex, reusing cached chunk vectors when valid."""
    fp = _fingerprint(chunks, EMBED_MODEL)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if str(data["fingerprint"]) == fp:
            print(f"Loaded cached vectors from {cache_path.name}")
            return DenseIndex(chunks, matrix=data["matrix"])
    index = DenseIndex(chunks)
    np.savez(cache_path, matrix=index.matrix, fingerprint=fp)
    print(f"Computed + cached vectors to {cache_path.name}")
    return index


# ------------------------------------------------------------------------ main

def answer(index, question):
    retrieved = index.retrieve(question)
    print("\nRetrieved chunks:")
    for score, src, text in retrieved:
        print(f"  {score:.3f}  {src}  \"{text[:70]}...\"")

    result, reason = generate(build_prompt(question, retrieved))
    if result is None:
        print(f"\n[{reason} — showing retrieved context only]")
        for _, src, text in retrieved:
            print(f"\n--- {src} ---\n{text}")
    else:
        print(f"\nAnswer: {result}")

def main():
    args = sys.argv[1:]
    use_tfidf = "--tfidf" in args          # default: dense embeddings
    if use_tfidf:
        args.remove("--tfidf")

    chunks = load_and_chunk()
    if not chunks:
        sys.exit(f"No .md files found in {DOCS_DIR}")

    nfiles = len(set(s for s, _ in chunks))
    if use_tfidf:
        index = TfidfIndex(chunks)
        print(f"TF-IDF index: {len(chunks)} chunks from {nfiles} files, vocab={len(index.vocab)}")
    else:
        index = load_dense_index(chunks)
        print(f"Dense index ({EMBED_MODEL}): {len(chunks)} chunks from {nfiles} files")

    if len(args) > 1 and args[0] == "--query":   # one-shot mode
        answer(index, args[1])
        return

    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            answer(index, question)

if __name__ == "__main__":
    main()
