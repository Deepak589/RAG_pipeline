#!/usr/bin/env python3
"""Synthetic QA labeler — turn a parent store into a versioned golden set.

The real Stage-5 cost is LABELING, not downloading: 1000 unlabeled docs give
zero eval signal because Recall@k needs gold questions. This generates them the
same disciplined way blurb_gen.py generates blurbs — deterministic Ollama calls,
fingerprinted cache — and emits qa.json in the EXACT schema run_eval.py already
consumes (factual / paraphrase / negative buckets, `relevant_parent_ids`, a
top-level `version`).

It works against a sections.json parent store (the same shape every stage uses:
records with source, section_idx, title, text). So the flow is:

    corpus_puller.py  -> raw docs
    <your chunker>    -> sections.json (parents) + chunks.json (children)
    qa_gen.py         -> qa.json  (THIS FILE)
    run_eval.py       -> numbers

Buckets:
  factual    — question answerable ONLY from one parent; that parent is the label
  paraphrase — the factual question reworded (tests dense > lexical); same label
  negative   — on-domain question whose answer is NOT in the corpus; label = []

DISCIPLINE (matches the project's own rules):
  * temperature=0, fixed seed, prompt version -> reproducible questions.
  * fingerprinted cache keyed on (prompt-ver | model | parent-id | parent-text)
    so a prompt/model change regenerates, an interrupted run resumes for free.
  * The output is a DRAFT golden set. Synthetic labels MUST be spot-checked by
    hand before you trust an eval number on them — generation can mislabel or
    write a "factual" question that other parents also answer. Bump the version
    string every time you regenerate.

Run (Ollama up):
    python qa_gen.py --sections RAG/rag_stage_4/sections.json --out qa_v3.json \
                     --factual 120 --paraphrase 40 --negative 20 --version v3-...
    python qa_gen.py --sections ... --mock          # templated, no LLM (plumbing)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"          # was "qwen3.5" — not a real Ollama tag (crash)
SEED = 7
PROMPT_VERSION = "v1"
PARENT_CHAR_BUDGET = 3500
MIN_PARENT_CHARS = 400          # skip stubs too thin to ask a real question about
SKIP_TITLE_RE = re.compile(r"content|reference|bibliograph|acknowledg|appendix",
                           re.I)
MIN_SPACE_RATIO = 0.02          # below this it's XBRL-tag soup, not prose
RETRY_ATTEMPTS = 4              # bump seed and retry on dup/garbage LLM output


# ------------------------------------------------------------------ ollama call

def call_ollama(prompt, model, seed=SEED, temperature=0):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "format": "json",          # force valid JSON out -> kills parse_json_obj misses
        "options": {"temperature": temperature, "seed": seed},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"].strip()


def parse_json_obj(text):
    """Pull the first {...} JSON object out of a possibly chatty response."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------- prompts

def factual_prompt(paper, title, text):
    return (
        "From the passage below, write ONE specific factual question that can be "
        "answered using ONLY this passage, and give its short answer. The "
        "question must name enough detail that it is NOT answerable from a "
        "generic passage on the same topic. Reply as strict JSON on one line: "
        '{"question": "...", "answer": "..."}\n\n'
        f"Paper: {paper}\nSection: {title}\n\nPassage:\n{text[:PARENT_CHAR_BUDGET]}\n\n"
        "JSON:"
    )


def is_clean_text(text, max_non_ascii_ratio=0.2):
    """Reject LLM output that wandered into another script or dropped in
    invisible/format unicode chars (seen: CJK output, U+2062 INVISIBLE TIMES)."""
    if any(unicodedata.category(ch) == "Cf" for ch in text):
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / max(1, len(text)) <= max_non_ascii_ratio


def paraphrase_prompt(question):
    return (
        "Reword this question so it keeps the exact same meaning and answer but "
        "shares as few content words as possible (use synonyms, change the "
        'structure). Reply as strict JSON on one line: {"question": "..."}\n\n'
        f"Original: {question}\n\nJSON:"
    )


def negative_prompt(domain_titles):
    joined = "; ".join(domain_titles[:12])
    return (
        "Here are section titles from a document corpus:\n"
        f"{joined}\n\n"
        "Write ONE question that sounds like it belongs to this domain but whose "
        "answer is almost certainly NOT contained in such documents (too "
        "specific, too recent, or about an entity not covered). Reply as strict "
        'JSON on one line: {"question": "..."}\n\nJSON:'
    )


# ------------------------------------------------------------------------ cache

def fingerprint(model, kind, pid, text):
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, model, kind, pid, text):
        h.update(part.encode()); h.update(b"\0")
    return h.hexdigest()


def short_source(source):
    parts = source.split(" - ")
    return parts[0].strip() if parts else source


def slug(source):
    return re.sub(r"[^a-z0-9]+", "-", short_source(source).lower()).strip("-")[:24]


# --------------------------------------------------------------------- validate

_WORD_RE = re.compile(r"[a-z0-9]+")
# tiny stoplist so the negative-overlap check keys on CONTENT words, not glue
_STOP = set("the a an of to in for on and or is are was were be been being this that "
            "these those with as by at from it its their his her what which who how "
            "why when where does do did can could would should will may might has have "
            "had not no than then into about over under between".split())


def _content_tokens(s):
    return {t for t in _WORD_RE.findall((s or "").lower()) if t not in _STOP}


def load_parent_map(path):
    secs = json.loads(Path(path).read_text())
    return {f"{s['source']}#{s['section_idx']}": s for s in secs}


def validate(args):
    """Quality gate on a generated qa.json. Catches the mislabels qa_gen can't
    see itself, using cheap checks + the SAME retriever that will grade:

      1. groundedness (factual) — does _answer's content actually live in the
         labeled parent? low token overlap => hallucinated question. (no retriever)
      2. multi-label (factual)  — retrieve the query; if a NON-gold parent outranks
         the gold one, the answer likely lives in several parents => suggest adding
         them (fixes single-label recall understatement). (needs retriever)
      3. negative               — max content-word overlap between the negative
         query and ANY parent; high overlap => probably answerable => bad negative.

    Writes <out>.validated.json with a `_validate` note per flagged question and
    prints a report. It FLAGS for human review, does not silently rewrite labels —
    and it only catches labels the retriever DISAGREES with, so hand-verify the
    flagged set; it's a net, not a wall."""
    data = json.loads(Path(args.out).read_text())
    qs = data["questions"]
    parents = load_parent_map(args.sections)

    retr = None
    if not args.no_retriever:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag_stage_5"))
            import lc_pipeline as lc
            children = lc.load_children(args.chunks)
            emb = lc.make_embeddings(args.embeddings, args.embed_model)
            retr = lc.build_hybrid(children, emb, vector_store=args.vector_store,
                                   conn=args.conn, collection=args.collection)
            print(f"validate: retriever up ({args.vector_store}), {len(children)} children")
        except Exception as e:
            print(f"validate: NO retriever ({str(e)[:80]}) — skipping multi-label "
                  "check (groundedness + negative still run)")

    flagged = 0
    for q in qs:
        notes = []
        if q["type"] == "factual":
            gold = (q.get("relevant_parent_ids") or [None])[0]
            ans = _content_tokens(q.get("_answer", ""))
            if ans and gold in parents:
                cov = len(ans & _content_tokens(parents[gold]["text"])) / len(ans)
                if cov < args.ground_min:
                    notes.append(f"low-groundedness (answer overlap {cov:.2f} < "
                                 f"{args.ground_min}) — question may be ungrounded")
            if retr is not None:
                ids, _ = lc.rank_parents(retr, q["query"])
                gset = set(q.get("relevant_parent_ids") or [])
                topk = ids[:args.topk]
                if ids and ids[0] not in gset:
                    cand = [p for p in topk if p not in gset][:3]
                    notes.append(f"gold not rank-1 (rank-1={ids[0]}); "
                                 f"consider multi-label: {cand}")
                elif not (gset & set(topk)):
                    notes.append(f"gold absent from top-{args.topk} — bad label or hard Q")
        elif q["type"] == "negative":
            qt = _content_tokens(q["query"])
            if qt:
                best = max((len(qt & _content_tokens(p["text"])) / len(qt)
                            for p in parents.values()), default=0.0)
                if best > args.neg_max:
                    notes.append(f"possibly ANSWERABLE (content overlap {best:.2f} > "
                                 f"{args.neg_max}) — weak negative")
        if notes:
            q["_validate"] = notes
            flagged += 1

    out = Path(args.out).with_suffix(".validated.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    kinds = Counter(n.split(" ")[0] + (" " + n.split(" ")[1] if len(n.split()) > 1 else "")
                    for q in qs for n in q.get("_validate", []))
    print(f"\nvalidate: {flagged}/{len(qs)} questions flagged -> {out.name}")
    for k, c in kinds.most_common():
        print(f"    {c:>3}  {k}")
    print("HAND-VERIFY the flagged questions; apply multi-label suggestions you agree "
          "with, drop weak negatives/ungrounded factuals, then bump the version.")


# ------------------------------------------------------------------------- main

def load_parents(path):
    secs = json.loads(Path(path).read_text())
    parents = []
    for s in secs:
        pid = f"{s['source']}#{s['section_idx']}"
        text = s.get("text", "")
        if len(text) < MIN_PARENT_CHARS:
            continue
        if SKIP_TITLE_RE.search(s.get("title", "")):
            continue
        if text.count(" ") / len(text) < MIN_SPACE_RATIO:
            continue  # raw XBRL fact dump (concatenated tags, no prose)
        parents.append({"pid": pid, "source": s["source"],
                        "title": s.get("title", ""), "text": text})
    return parents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True, help="parent store (sections.json)")
    ap.add_argument("--out", default="qa_generated.json")
    ap.add_argument("--factual", type=int, default=120)
    ap.add_argument("--paraphrase", type=int, default=40)
    ap.add_argument("--negative", type=int, default=20)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--version", default=None, help="version stamp for qa.json")
    ap.add_argument("--mock", action="store_true", help="templated, no LLM")
    # ---- validate mode (post-generation quality gate) ----
    ap.add_argument("--validate", action="store_true",
                    help="check an existing --out qa.json instead of generating")
    ap.add_argument("--chunks", help="chunks.json for the multi-label retriever")
    ap.add_argument("--ground-min", type=float, default=0.5,
                    help="min answer->parent token overlap for a factual (else flag)")
    ap.add_argument("--neg-max", type=float, default=0.6,
                    help="max query->parent overlap for a negative (else flag)")
    ap.add_argument("--topk", type=int, default=5, help="multi-label lookahead depth")
    ap.add_argument("--no-retriever", action="store_true",
                    help="skip the retriever-based multi-label check")
    ap.add_argument("--embeddings", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--vector-store", choices=["memory", "pgvector"], default="memory")
    ap.add_argument("--conn", default=os.environ.get(
        "PG_CONN", "postgresql+psycopg://postgres:postgres@localhost:5432/ragdev"))
    ap.add_argument("--collection", default="rag_stage5")
    args = ap.parse_args()

    if args.validate:
        validate(args)
        return

    parents = load_parents(args.sections)
    if not parents:
        sys.exit("no usable parents (all too short or filtered) — check --sections")
    print(f"{len(parents)} candidate parents "
          f"(>= {MIN_PARENT_CHARS} chars, non-boilerplate)")

    cache_path = Path(args.out).with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    model_id = "MOCK" if args.mock else args.model

    def cached(kind, pid, text, make):
        fp = fingerprint(model_id, kind, pid, text)
        hit = cache.get(f"{kind}:{pid}")
        if hit and hit.get("fp") == fp:
            return hit["val"]
        val = make()
        cache[f"{kind}:{pid}"] = {"fp": fp, "val": val}
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        return val

    questions = []
    used = set()

    # ---- factual (spread across distinct parents) ----
    n_fact = min(args.factual, len(parents))
    step = max(1, len(parents) // n_fact)
    picked = parents[::step][:n_fact]
    for i, p in enumerate(picked):
        def make(p=p):
            if args.mock:
                first = " ".join(p["text"].split()[:6])
                return {"question": f"What does the section '{p['title']}' say about "
                        f"{first}?", "answer": first}
            obj = parse_json_obj(call_ollama(
                factual_prompt(short_source(p["source"]), p["title"], p["text"]),
                args.model))
            return obj or {}
        try:
            obj = cached("factual", p["pid"], p["text"], make)
        except (urllib.error.URLError, OSError) as e:
            sys.exit(f"Ollama failed: {e}\nProgress cached in {cache_path.name}; "
                     "fix Ollama and re-run to resume, or use --mock.")
        q = (obj or {}).get("question", "").strip()
        if not q:
            continue
        qid = f"{slug(p['source'])}-{p['pid'].split('#')[-1]}-f"
        questions.append({"id": qid, "type": "factual", "query": q,
                          "relevant_parent_ids": [p["pid"]],
                          "_answer": obj.get("answer", ""), "_gen": True})
        used.add(qid)
        if (i + 1) % 10 == 0:
            print(f"  factual {i+1}/{len(picked)}")

    # ---- paraphrase (reword a subset of the factual questions) ----
    # spread across the whole factual set (was fact_qs[:N] — clustered on the
    # alphabetically-first sources).
    fact_qs = [q for q in questions if q["type"] == "factual"]
    p_step = max(1, len(fact_qs) // max(1, args.paraphrase))
    for q in fact_qs[::p_step][:args.paraphrase]:
        pq = None
        for attempt in range(RETRY_ATTEMPTS):
            def make(q=q, attempt=attempt):
                if args.mock:
                    return {"question": "Reworded: " + q["query"]}
                return parse_json_obj(call_ollama(
                    paraphrase_prompt(q["query"]), args.model,
                    seed=SEED + attempt,
                    temperature=0 if attempt == 0 else 0.8)) or {}
            pid = q["relevant_parent_ids"][0] + (f"-r{attempt}" if attempt else "")
            obj = cached("paraphrase", pid, q["query"], make)
            cand = (obj or {}).get("question", "").strip()
            if cand and is_clean_text(cand):
                pq = cand
                break
        if not pq:
            continue
        questions.append({"id": q["id"].rsplit("-", 1)[0] + "-p",
                          "type": "paraphrase", "query": pq,
                          "relevant_parent_ids": list(q["relevant_parent_ids"]),
                          "_gen": True})

    # ---- negatives (on-domain, answer absent) ----
    titles = [p["title"] for p in parents if p["title"]]
    seen_negatives = set()
    for j in range(args.negative):
        # vary the title window per index so negatives aren't identical
        window = titles[j % max(1, len(titles) - 12): j % max(1, len(titles) - 12) + 12] or titles
        nq = None
        for attempt in range(RETRY_ATTEMPTS):
            def make(window=window, j=j, attempt=attempt):
                if args.mock:
                    return {"question": f"Mock negative #{j}-{attempt} about an "
                            "uncovered topic?"}
                return parse_json_obj(call_ollama(
                    negative_prompt(window), args.model, seed=SEED + attempt,
                    temperature=0 if attempt == 0 else 0.8)) or {}
            pid = f"neg{j}" + (f"-r{attempt}" if attempt else "")
            obj = cached("negative", pid, " ".join(window), make)
            cand = (obj or {}).get("question", "").strip()
            if cand and cand.lower() not in seen_negatives:
                nq = cand
                break
        if not nq:
            continue  # exhausted retries on duplicates — drop rather than repeat
        seen_negatives.add(nq.lower())
        questions.append({"id": f"neg-{j}", "type": "negative", "query": nq,
                          "relevant_parent_ids": [], "_gen": True})

    version = args.version or f"gen-{len(questions)}q-{PROMPT_VERSION}"
    out = {"version": version, "questions": questions}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    from collections import Counter
    by = Counter(q["type"] for q in questions)
    print(f"\nwrote {len(questions)} questions -> {args.out}  (version {version})")
    print(f"  factual {by['factual']}  paraphrase {by['paraphrase']}  "
          f"negative {by['negative']}")
    print("REMINDER: synthetic labels are a DRAFT — hand-verify a sample before "
          "trusting eval numbers, and drop the `_answer`/`_gen` helper fields "
          "once validated.")


if __name__ == "__main__":
    main()
