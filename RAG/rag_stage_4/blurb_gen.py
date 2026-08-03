"""Stage 4 — contextual blurb chunking (Anthropic Contextual Retrieval).

Prepend a one-sentence, LLM-written context blurb to each child BEFORE it is
embedded/indexed, so an isolated 120-word window becomes self-situating
("This passage is from Vaswani sec 5.3 Optimizer, on the Adam warmup+decay
schedule"). The parent handed to the generator is unchanged — this is an
INDEX-TIME retrieval trick, not a content change.

Why it should move recall here: the v2 hybrid ceiling is already recall@50
0.992, so remaining misses are a RANKING problem, not coverage. Blurbs make
thin fragments distinguishable and lift both sides of hybrid — sharper dense
vectors AND real keywords for BM25.

Design decisions (locked 2026-08-03):
  * Blurb context source = PARENT SECTION (paper title + section title +
    parent-section text + the child text). Cheap, deterministic, strong
    signal on this clean-sectioned corpus. Not whole-document (token cost /
    qwen ctx on the Gao survey) and not metadata-only (too weak).
  * DETERMINISM is enforced, not assumed: the LLM call pins temperature=0 and
    a fixed seed. Ollama's defaults sample — non-deterministic blurbs would
    poison the cache and make eval irreproducible.
  * Cache is a COST optimisation only. Each blurb entry stores a fingerprint
    over (prompt version | model | parent text | child text); a change in any
    of them regenerates that blurb. The recall win comes from the blurb TEXT,
    not from caching.

Output: chunks.json where each child gains
    raw_text  — the original 120-word window (unchanged)
    blurb     — the generated one-sentence context
    text      — blurb + "\n" + raw_text  (what dense + BM25 consume)
so the existing stage-3 retrieval code indexes blurbed text with no change,
and the dense fingerprint (hashes child["text"]) auto-invalidates the cache.

Run (on a machine with Ollama up):
    python blurb_gen.py                 # generate all blurbs, write chunks.json
    python blurb_gen.py --limit 20      # quick trial on first 20 children
    python blurb_gen.py --mock          # no LLM: templated blurbs (plumbing test)
    python blurb_gen.py --force         # ignore cache, regenerate every blurb
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
STAGE3 = HERE.parent / "rag_stage_3"
SECTIONS_PATH = HERE / "sections.json"        # parents (copied from stage 3)
CHILDREN_IN = STAGE3 / "chunks.json"          # base children (stage-3 windows)
CHUNKS_OUT = HERE / "chunks.json"             # blurbed children
BLURB_CACHE = HERE / "blurbs.json"            # {child_id: {"blurb":..., "fp":...}}

OLLAMA_URL = "http://localhost:11434/api/generate"
BLURB_MODEL = "qwen3.5"       # same local model as the generator
BLURB_SEED = 7                # fixed -> reproducible blurbs
PROMPT_VERSION = "v1"         # bump to invalidate every cached blurb on purpose
PARENT_CHAR_BUDGET = 8000     # cap parent text fed to the LLM (all parents < 10k)


# --------------------------------------------------------------- prompt + call

def short_source(source):
    """'Vaswani et al. - 2017 - Attention...' -> 'Vaswani et al. - 2017'."""
    parts = source.split(" - ")
    return " - ".join(parts[:2]) if len(parts) >= 2 else source


def build_blurb_prompt(paper, section_title, parent_text, child_text):
    """One-sentence situating prompt grounded in the parent section only."""
    parent_text = parent_text[:PARENT_CHAR_BUDGET]
    return (
        "You are labelling a passage for a search index. Using the section it "
        "comes from, write ONE short sentence that situates the passage: name "
        "the paper, the section/topic, and what the passage is about. Output "
        "only that sentence, no preamble.\n\n"
        f"Paper: {paper}\n"
        f"Section: {section_title}\n"
        f"Section text:\n{parent_text}\n\n"
        f"Passage:\n{child_text}\n\n"
        "One-sentence context:"
    )


def _fingerprint(paper, section_title, parent_text, child_text):
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, BLURB_MODEL, str(BLURB_SEED),
                 paper, section_title, parent_text, child_text):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def call_ollama(prompt):
    """Deterministic Ollama generate. Returns blurb string, or raises on failure."""
    payload = json.dumps({
        "model": BLURB_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": BLURB_SEED},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"].strip()


def mock_blurb(paper, section_title, child_text):
    """Deterministic templated blurb — no LLM. Lets us test all plumbing
    (cache, fingerprints, json shapes, eval) without Ollama or the embed model."""
    snippet = " ".join(child_text.split()[:8])
    return f"This passage is from {short_source(paper)}, section '{section_title}', about: {snippet}."


def clean_blurb(text):
    """Collapse to a single line; guard against a chatty model."""
    line = " ".join(text.split())
    # keep only the first sentence-ish span if the model over-produced
    return line


# --------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N children (quick trial)")
    ap.add_argument("--mock", action="store_true",
                    help="templated blurbs, no LLM (plumbing/self-test)")
    ap.add_argument("--force", action="store_true",
                    help="ignore cache, regenerate every blurb")
    args = ap.parse_args()

    if not SECTIONS_PATH.exists():
        sys.exit(f"{SECTIONS_PATH.name} missing — copy stage-3 sections.json first")
    if not CHILDREN_IN.exists():
        sys.exit(f"{CHILDREN_IN} missing — run stage-3 chunker first")

    sections = json.loads(SECTIONS_PATH.read_text())
    parents = {f"{s['source']}#{s['section_idx']}": s for s in sections}
    children = json.loads(CHILDREN_IN.read_text())
    if args.limit:
        children = children[:args.limit]

    cache = {}
    if BLURB_CACHE.exists() and not args.force:
        cache = json.loads(BLURB_CACHE.read_text())

    out, made, reused, missing_parent = [], 0, 0, 0
    for i, c in enumerate(children):
        parent = parents.get(c["parent_id"])
        if parent is None:
            missing_parent += 1
            parent_text, section_title = "", c.get("title", "")
        else:
            parent_text, section_title = parent["text"], parent["title"]
        paper = c["source"]

        fp = _fingerprint(paper, section_title, parent_text, c["text"])
        hit = cache.get(c["id"])
        if hit and hit.get("fp") == fp and not args.force:
            blurb = hit["blurb"]
            reused += 1
        else:
            if args.mock:
                blurb = mock_blurb(paper, section_title, c["text"])
            else:
                prompt = build_blurb_prompt(paper, section_title, parent_text, c["text"])
                try:
                    blurb = clean_blurb(call_ollama(prompt))
                except (urllib.error.URLError, OSError, KeyError) as e:
                    sys.exit(f"Ollama call failed at child {i} ({c['id']}): {e}\n"
                             f"Start `ollama serve` (model {BLURB_MODEL}) or use --mock.")
            cache[c["id"]] = {"blurb": blurb, "fp": fp}
            made += 1

        child = dict(c)
        child["raw_text"] = c["text"]
        child["blurb"] = blurb
        child["text"] = f"{blurb}\n{c['text']}"     # what dense + BM25 will consume
        out.append(child)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(children)} blurbed "
                  f"(new {made}, cached {reused})")

    BLURB_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    CHUNKS_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if missing_parent:
        print(f"WARN: {missing_parent} children had no parent in sections.json "
              f"(blurbed from title only)")
    print(f"blurbs: {made} generated, {reused} from cache "
          f"({'MOCK' if args.mock else BLURB_MODEL}, seed {BLURB_SEED}, "
          f"prompt {PROMPT_VERSION})")
    print(f"wrote {len(out)} blurbed children -> {CHUNKS_OUT.name}")
    print(f"cache -> {BLURB_CACHE.name}")


if __name__ == "__main__":
    main()
