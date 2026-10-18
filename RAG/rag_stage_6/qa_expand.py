#!/usr/bin/env python3
"""Grow the golden set on the 40pct corpus — more positives = less noise per Q.

At 141 positives one question is ~0.7 pts; doubling the set halves that. Reuses
qa_gen.py's prompts + parent filters (same question style as v3) but calls Claude
instead of a local Ollama model, and only labels parents that are IN the 40pct
subset and NOT already gold in v3. Factual sources are stratified to match v3's
mix so the merged set keeps the same shape. Negatives are NEAR-MISS (same entity
as a real passage, absent detail), verified unanswerable against the retriever's
top-5 — v3's off-domain negatives make correct_refusal trivially 100%.

Output = v3 + new questions, ready for crag_eval.py --qa. New questions keep
`_gen: "claude-expand"` so they can be sliced out or dropped. Run qa_gen.py
--validate on the output afterwards (groundedness / negative-overlap flags).

    python qa_expand.py --factual 100 --paraphrase 40 --negative 20 \
        --out qa_stage6_v4.json
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

import crag_pipeline as crag   # puts rag_stage_5 on sys.path
from crag_pipeline import s5

sys.path.insert(0, str(crag.S5_DIR))
import qa_gen  # noqa: E402

SEED = 7
GROUND_MIN = 0.5      # same bar as qa_gen --validate --ground-min
PARA_MAX = 0.6        # max content-word overlap paraphrase<->original


class QA(BaseModel):
    question: str
    answer: str


class Q(BaseModel):
    question: str


class Answerable(BaseModel):
    answerable: bool = Field(description="true if ANY document states the answer")


def overlap(a, b):
    ta, tb = qa_gen._content_tokens(a), qa_gen._content_tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


# qa_gen's prompt alone yielded "refund liabilities as of Dec 31, 2025?" (which
# company?) and "the passage from sec/0000064040-..." -> unanswerable / leaky.
FACT_RULES = ("\nRules: name the specific entity (company, paper, product, person) "
              "by its real name — never 'the Company', 'this filing', 'the passage', "
              "'the document', 'this exhibit', or file/accession/CIK IDs; if the "
              "passage never names the entity, ask about something it does name. "
              "Don't ask about bibliography entries.")


# Hard gate behind the prompt rules: the first full run still produced 29/160
# "According to the passage ..." / "this SEC filing (accession ...)" questions,
# and paraphrases that were the prompt text or raw JSON echoed back.
LEAKY_RE = re.compile(r"\b(the|this) (passage|document|filing|exhibit|text|section|"
                      r"sec filing|company)\b|accession|\bCIK\b|sec/\d|arxiv/\d|"
                      r"original:|reword|rewrite|^\s*\{", re.I)


def ok_question(q):
    return bool(q) and len(q) < 300 and qa_gen.is_clean_text(q) and not LEAKY_RE.search(q)


def paraphrase_prompt(question):
    return ("Ask for exactly the same fact as the question below, worded the way "
            "a different person would ask it: use synonyms and a different sentence "
            "structure, but keep proper names. Return only the new question.\n\n"
            f"Question: {question}")


def near_miss_prompt(p):
    return (
        "Write ONE question about the same specific entity (company, paper, "
        "project, person) as the passage below, asking for a concrete detail "
        "(a figure, date, name, or fact) that the passage does NOT contain and "
        "that a reader would plausibly expect to find in documents like it. "
        + FACT_RULES + "\n\n"
        f"Source: {qa_gen.short_source(p['source'])}\nSection: {p['title']}\n\n"
        f"Passage:\n{p['text'][:qa_gen.PARENT_CHAR_BUDGET]}")


def grounded(answer, text):
    a = qa_gen._content_tokens(answer)
    return bool(a) and len(a & qa_gen._content_tokens(text)) / len(a) >= GROUND_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="subset_40pct/sections.json")
    ap.add_argument("--base", default=str(crag.S5_DIR / "qa_stage5_v3.json"))
    ap.add_argument("--out", default="qa_stage6_v4.json")
    ap.add_argument("--factual", type=int, default=100)
    ap.add_argument("--paraphrase", type=int, default=40)
    ap.add_argument("--negative", type=int, default=20)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--check-model", default="claude-sonnet-5",
                    help="verifies near-miss negatives are truly unanswerable")
    ap.add_argument("--conn", default="postgresql+psycopg://deepakkatukuri@localhost:5432/ragdev_40pct")
    ap.add_argument("--collection", default="rag_stage5__bge_m3__40pct")
    ap.add_argument("--workers", type=int, default=4)
    crag.add_guard_args(ap)
    args = ap.parse_args()
    crag.enable_guards(args)

    base = json.loads(Path(args.base).read_text())
    gold = {p for q in base["questions"] for p in q["relevant_parent_ids"]}
    rng = random.Random(SEED)

    # ---- pick parents: unlabeled, stratified to v3's factual source mix --------
    by_src = {}
    for p in qa_gen.load_parents(args.sections):
        if p["pid"] not in gold:
            by_src.setdefault(p["source"].split("/")[0], []).append(p)
    mix = Counter(q["relevant_parent_ids"][0].split("/")[0]
                  for q in base["questions"] if q["type"] == "factual")
    total = sum(mix.values())
    picked = []
    for src, c in mix.items():
        pool = by_src.get(src, [])
        # oversample 1.7x: ~1/4 fail groundedness, ~1/5 fail the leak gate
        picked += rng.sample(pool, min(len(pool), round(1.7 * args.factual * c / total)))

    llm = crag.make_chat_anthropic(args.model, temperature=0)
    qa_llm = llm.with_structured_output(QA)
    # Haiku echoed the original verbatim on paraphrase; Sonnet rewords properly
    q_llm = crag.make_chat_anthropic(args.check_model, temperature=0
                                     ).with_structured_output(Q)

    def make_factual(p):
        try:
            # rules FIRST: appended after qa_gen's trailing "JSON:" they were ignored
            r = qa_llm.invoke(FACT_RULES.strip() + "\n\n" + qa_gen.factual_prompt(
                qa_gen.short_source(p["source"]), p["title"], p["text"]))
        except Exception as e:          # one bad call shouldn't sink the batch
            print(f"  skip {p['pid']}: {str(e)[:80]}")
            return None
        if not (ok_question(r.question) and grounded(r.answer, p["text"])):
            return None
        return {"id": f"{qa_gen.slug(p['source'])}-{p['pid'].split('#')[-1]}-x-f",
                "type": "factual", "query": r.question.strip(),
                "relevant_parent_ids": [p["pid"]], "_answer": r.answer.strip(),
                "_gen": "claude-expand"}

    with ThreadPoolExecutor(args.workers) as ex:
        facts = [f for f in ex.map(make_factual, picked) if f]
    # trim back to the target while keeping the stratified mix
    rng.shuffle(facts)
    facts = facts[:args.factual]
    print(f"factual: {len(facts)} kept of {len(picked)} tried")

    def make_para(f):
        # reject echoes of the original and ask again with that feedback
        prompt = paraphrase_prompt(f["query"])
        for _ in range(3):
            try:
                q = q_llm.invoke(prompt).question.strip()
            except Exception as e:
                print(f"  skip para {f['id']}: {str(e)[:80]}")
                return None
            if ok_question(q) and overlap(q, f["query"]) <= PARA_MAX:
                return {"id": f["id"][:-2] + "-p", "type": "paraphrase", "query": q,
                        "relevant_parent_ids": list(f["relevant_parent_ids"]),
                        "_gen": "claude-expand"}
            prompt += f"\n\nToo similar: {q!r}. Use different words and structure."
        return None

    with ThreadPoolExecutor(args.workers) as ex:
        paras = [p for p in ex.map(make_para, facts[:args.paraphrase]) if p]
    print(f"paraphrase: {len(paras)} kept of {min(len(facts), args.paraphrase)}")

    # ---- near-miss negatives -------------------------------------------------
    # v3 negatives are off-domain trivia (weather in Beijing) -> 100% refusal is
    # free. Near-miss = same entity as a real passage, asking a detail it lacks.
    # Kept only if neither the source nor any retrieved top-5 parent answers it.
    tried = {p["pid"] for p in picked}
    pool = [p for ps in by_src.values() for p in ps if p["pid"] not in tried]
    neg_src = rng.sample(pool, round(args.negative * 3))   # ~half survive the check
    parents = s5.load_parents(args.sections)
    retr = s5.build_hybrid([], s5.make_embeddings("hf", "BAAI/bge-m3"),
                           vector_store="pgvector", conn=args.conn,
                           collection=args.collection)
    checker = crag.make_chat_anthropic(args.check_model, temperature=0
                                       ).with_structured_output(Answerable)

    def make_neg(p):
        try:
            q = q_llm.invoke(near_miss_prompt(p)).question.strip()
            if not ok_question(q):
                return None
            pids = [p["pid"]] + [x for x in s5.rank_parents(retr, q)[0][:5] if x != p["pid"]]
            ctx = "\n\n".join(f"[{x}]\n{parents[x]['text'][:3000]}" for x in pids)
            if checker.invoke(f"Question: {q}\n\nDocuments:\n{ctx}").answerable:
                print(f"  reject neg (answerable): {q[:100]}")
                return None
        except Exception as e:
            print(f"  skip neg {p['pid']}: {str(e)[:80]}")
            return None
        return {"id": f"neg-x-{qa_gen.slug(p['source'])}-{p['pid'].split('#')[-1]}",
                "type": "negative", "query": q, "relevant_parent_ids": [],
                "_near_miss_of": p["pid"], "_gen": "claude-expand"}

    with ThreadPoolExecutor(args.workers) as ex:
        negs = [n for n in ex.map(make_neg, neg_src) if n][:args.negative]
    print(f"negative: {len(negs)} kept of {len(neg_src)} tried")

    new = facts + paras + negs
    out = {"version": f"stage6_v4-40pct-{len(base['questions']) + len(new)}q",
           "questions": base["questions"] + new}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}: +{len(facts)} factual +{len(paras)} paraphrase "
          f"+{len(negs)} negative  -> {Counter(q['type'] for q in out['questions'])}")


if __name__ == "__main__":
    main()
