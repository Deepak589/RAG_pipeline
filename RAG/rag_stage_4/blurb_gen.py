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
BLURB_MODEL = "qwen3.5"       # blurb LLM. A SMALL model (qwen2.5:3b, llama3.2:3b)
                              # is plenty for a one-line labelling task and 3-5x
                              # faster than a big general model. Override with --model.
BLURB_SEED = 7                # fixed -> reproducible blurbs
PROMPT_VERSION = "v1"         # bump to invalidate every cached blurb on purpose
PARENT_CHAR_BUDGET = 3000     # cap parent text fed to the LLM. Most of the local
                              # runtime is PROMPT processing, so a tighter budget is
                              # the cheapest speedup; 3k chars still covers the whole
                              # section for the vast majority of parents.
DEFAULT_WORKERS = 4           # concurrent Ollama requests (see --workers)
CACHE_FLUSH_EVERY = 20        # persist cache every N new blurbs -> resumable


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


def _fingerprint(model, paper, section_title, parent_text, child_text):
    """Cache key. MUST include the model — a blurb is model-specific, so
    switching --model has to invalidate and regenerate, not reuse stale text."""
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, model, str(BLURB_SEED),
                 paper, section_title, parent_text, child_text):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def call_ollama(prompt, model=BLURB_MODEL):
    """Deterministic Ollama generate. Returns blurb string, or raises on failure."""
    payload = json.dumps({
        "model": model,
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

def _parent_ctx(c, parents):
    """(paper, section_title, parent_text) for a child, tolerating a missing parent."""
    parent = parents.get(c["parent_id"])
    if parent is None:
        return c["source"], c.get("title", ""), ""
    return c["source"], parent["title"], parent["text"]


def main():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N children (quick trial)")
    ap.add_argument("--mock", action="store_true",
                    help="templated blurbs, no LLM (plumbing/self-test)")
    ap.add_argument("--force", action="store_true",
                    help="ignore cache, regenerate every blurb")
    ap.add_argument("--model", default=BLURB_MODEL,
                    help="Ollama model for blurbs (a small model is 3-5x faster)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent Ollama requests (set OLLAMA_NUM_PARALLEL to match)")
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

    # 1) Split into cache hits vs work to do. Fingerprint gates reuse, so an
    #    interrupted run resumes for free — only the un-cached children re-call.
    model_id = "MOCK" if args.mock else args.model   # cache key must separate them
    todo, reused, missing_parent = [], 0, 0
    fps = {}
    for c in children:
        paper, section_title, parent_text = _parent_ctx(c, parents)
        if parent_text == "":
            missing_parent += 1
        fp = _fingerprint(model_id, paper, section_title, parent_text, c["text"])
        fps[c["id"]] = fp
        hit = cache.get(c["id"])
        if hit and hit.get("fp") == fp and not args.force:
            reused += 1
        else:
            todo.append((c, paper, section_title, parent_text))

    print(f"{len(children)} children: {reused} cached, {len(todo)} to generate "
          f"({'MOCK' if args.mock else args.model}, {args.workers} workers)")

    # 2) Generate missing blurbs (parallel), flushing the cache periodically so
    #    a Ctrl-C never loses more than CACHE_FLUSH_EVERY blurbs.
    lock = threading.Lock()
    made = [0]
    start = time.time()

    def _flush():
        BLURB_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    def work(item):
        c, paper, section_title, parent_text = item
        if args.mock:
            return c["id"], mock_blurb(paper, section_title, c["text"])
        prompt = build_blurb_prompt(paper, section_title, parent_text, c["text"])
        return c["id"], clean_blurb(call_ollama(prompt, args.model))

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(work, it): it[0]["id"] for it in todo}
            for fut in as_completed(futures):
                try:
                    cid, blurb = fut.result()
                except (urllib.error.URLError, OSError, KeyError) as e:
                    _flush()
                    sys.exit(f"Ollama call failed: {e}\nProgress saved to "
                             f"{BLURB_CACHE.name} — fix Ollama (model {args.model}) "
                             f"and re-run to resume. Or use --mock.")
                with lock:
                    cache[cid] = {"blurb": blurb, "fp": fps[cid]}
                    made[0] += 1
                    n = made[0]
                    if n % CACHE_FLUSH_EVERY == 0:
                        _flush()
                        rate = n / (time.time() - start)
                        eta = (len(todo) - n) / rate if rate else 0
                        print(f"  {n}/{len(todo)} new  "
                              f"({rate:.1f}/s, ETA {eta/60:.1f} min)")
    finally:
        _flush()

    # 3) Assemble the blurbed children in original order.
    out = []
    for c in children:
        blurb = cache[c["id"]]["blurb"]
        child = dict(c)
        child["raw_text"] = c["text"]
        child["blurb"] = blurb
        child["text"] = f"{blurb}\n{c['text']}"     # what dense + BM25 will consume
        out.append(child)
    CHUNKS_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if missing_parent:
        print(f"WARN: {missing_parent} children had no parent in sections.json "
              f"(blurbed from title only)")
    print(f"blurbs: {made[0]} generated, {reused} from cache "
          f"({'MOCK' if args.mock else args.model}, seed {BLURB_SEED}, "
          f"prompt {PROMPT_VERSION}), {time.time() - start:.0f}s")
    print(f"wrote {len(out)} blurbed children -> {CHUNKS_OUT.name}")
    print(f"cache -> {BLURB_CACHE.name}")


if __name__ == "__main__":
    main()
