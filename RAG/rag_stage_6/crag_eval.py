#!/usr/bin/env python3
"""Stage 6 — CRAG eval harness: refusal + answer quality.

This is the automated gate that REPLACES manual one-knob eyeballing. Run it on
every change; the delta between MAX_STEPS=0 (single-shot control) and MAX_STEPS=2
(full CRAG) is the measured effect of the agentic layer.

Two metric families, measured on qa_stage5_v3.json (160 Q: 102 factual /
39 paraphrase / 19 negative):

  REFUSAL (the point of CRAG — open since Stage 3)
    correct_refusal   : % of the 19 NEGATIVES the system refused        (want HIGH)
    false_refusal     : % of POSITIVES the system wrongly refused        (want LOW)

  ANSWER QUALITY (Claude-as-judge on answered positives)
    answer_correct    : judge verdict correct / (correct+partial+wrong)  (want HIGH)
    also reports the retrieval->answer gap: gold retrieved but answer still wrong.

The judge is a SEPARATE Claude call scoring the generated answer against gold
(`_answer` when present, else the gold parent text). Judge is itself a knob;
temperature=0 and the rubric is fixed so runs are comparable.

Run (on your Mac):
    # full CRAG
    python crag_eval.py --qa qa_stage5_v3.json --max-steps 2 \
        --vector-store pgvector --collection rag_stage5__bge_m3 \
        --embeddings hf --embed-model BAAI/bge-m3 --out crag_eval_s2.json
    # single-shot control (the A/B baseline)
    python crag_eval.py --qa qa_stage5_v3.json --max-steps 0 --out crag_eval_s0.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import crag_pipeline as crag


class Judgement(BaseModel):
    verdict: Literal["correct", "partial", "wrong"] = Field(
        description="correct = fully answers and matches gold; partial = "
                    "on-topic but incomplete/imprecise; wrong = incorrect or unsupported")
    reason: str


def gold_answer(q, parents):
    """Ground truth for the judge: explicit _answer if present, else the gold
    parent text (paraphrase Qs have no _answer, only a gold parent)."""
    if q.get("_answer"):
        return q["_answer"]
    pids = q.get("relevant_parent_ids", [])
    return "\n".join(parents[p]["text"][:1500] for p in pids if p in parents)


def judge_answer(judge_llm, question, gold, answer) -> Judgement:
    prompt = (
        "Grade the ANSWER against the GOLD reference for the question. Be strict: "
        "an answer that is on-topic but omits the specific fact is 'partial'; an "
        "answer that states something the gold does not support is 'wrong'.\n\n"
        f"Question: {question}\nGold: {gold}\nAnswer: {answer}")
    return judge_llm.invoke(prompt)


def pct(n, d):
    return float("nan") if not d else round(100.0 * n / d, 1)


def evaluate(app, parents, judge_llm, qa_path, out_path):
    data = json.loads(Path(qa_path).read_text())
    qs = data["questions"]
    ver = data.get("version", "?")

    negatives = [q for q in qs if q["type"] == "negative"]
    positives = [q for q in qs if q["type"] in ("factual", "paraphrase")]

    rows = []
    t0 = time.time()

    # ---- negatives: did we correctly refuse? --------------------------------
    neg_refused = 0
    for i, q in enumerate(negatives, 1):
        r = crag.run_query(app, q["query"])
        if r["refused"]:
            neg_refused += 1
        rows.append({"id": q["id"], "type": "negative", "refused": r["refused"],
                     "steps_used": r["trace"].count("refine"), "trace": r["trace"]})
        print(f"  neg {i}/{len(negatives)}  refused={r['refused']}")

    # ---- positives: false-refusal + answer quality + retrieval attribution --
    # retrieval_hit = did a GOLD parent ever appear in the graded top-K (any
    # pass)? This splits a failure into its real cause instead of hiding a recall
    # drop inside the refusal count:
    #   retrieval_miss   gold never surfaced        -> index / retriever fault
    #   refused_on_hit   gold surfaced but refused  -> grader/generator too strict
    #   answered_wrong   gold surfaced, wrong answer -> generator fault
    #   correct          gold surfaced, right answer
    pos_false_refusal = 0
    retrieval_hits = 0
    judged = {"correct": 0, "partial": 0, "wrong": 0}
    attribution = {"retrieval_miss": 0, "refused_on_hit": 0,
                   "answered_wrong": 0, "correct": 0}
    for i, q in enumerate(positives, 1):
        r = crag.run_query(app, q["query"])
        gold = set(q.get("relevant_parent_ids", []))
        graded_pids = {g["parent_id"] for g in r.get("grades", [])}
        hit = bool(gold & graded_pids)
        retrieval_hits += int(hit)
        row = {"id": q["id"], "type": q["type"], "refused": r["refused"],
               "steps_used": r["trace"].count("refine"), "retrieval_hit": hit}
        if r["refused"]:
            pos_false_refusal += 1
            row["verdict"] = "refused"
            row["cause"] = "refused_on_hit" if hit else "retrieval_miss"
        else:
            j = judge_answer(judge_llm, q["query"], gold_answer(q, parents), r["answer"])
            judged[j.verdict] += 1
            row["verdict"] = j.verdict
            row["judge_reason"] = j.reason
            if j.verdict == "correct":
                row["cause"] = "correct"
            elif not hit:
                row["cause"] = "retrieval_miss"    # answered wrong AND gold never seen
            else:
                row["cause"] = "answered_wrong"
        attribution[row["cause"]] += 1
        rows.append(row)
        if i % 10 == 0:
            print(f"  pos {i}/{len(positives)}")

    elapsed = time.time() - t0
    n_pos = len(positives)
    answered = sum(judged.values())
    summary = {
        "version": ver,
        "config": {"max_steps": app_max_steps(app)},
        "counts": {"negatives": len(negatives), "positives": n_pos},
        "refusal": {
            "correct_refusal_pct": pct(neg_refused, len(negatives)),
            "false_refusal_pct": pct(pos_false_refusal, n_pos),
        },
        "answer_quality": {
            "answered": answered,
            "correct_pct": pct(judged["correct"], answered),
            "partial_pct": pct(judged["partial"], answered),
            "wrong_pct": pct(judged["wrong"], answered),
            "end_to_end_correct_pct": pct(judged["correct"], n_pos),  # incl. false refusals as non-correct
        },
        "retrieval": {
            # gold surfaced in the graded top-K -> isolates index/recall health
            # from grader+generator. A drop HERE is the HNSW/ef_search / FTS knob;
            # a drop in answer_quality with retrieval_hit high is a prompt knob.
            "retrieval_hit_pct": pct(retrieval_hits, n_pos),
            "attribution": attribution,   # where each positive was lost
        },
        "cost": {"wall_seconds": round(elapsed, 1),
                 "sec_per_q": round(elapsed / max(1, len(qs)), 2)},
    }

    Path(out_path).write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=1))

    _print_report(summary, out_path)
    return summary


def app_max_steps(app):
    # best-effort: pull the guard out of the compiled graph if exposed, else '?'
    return getattr(app, "_crag_max_steps", "?")


def _print_report(s, out_path):
    print("\n" + "=" * 68)
    print(f"CRAG eval — qa {s['version']}   max_steps={s['config']['max_steps']}")
    print("=" * 68)
    r, a = s["refusal"], s["answer_quality"]
    print(f"REFUSAL   correct-refusal (19 neg): {r['correct_refusal_pct']:>5}%   "
          f"false-refusal (pos): {r['false_refusal_pct']:>5}%")
    print(f"ANSWER    correct: {a['correct_pct']:>5}%   partial: {a['partial_pct']:>5}%"
          f"   wrong: {a['wrong_pct']:>5}%   (of {a['answered']} answered)")
    print(f"END-TO-END correct (of all positives): {a['end_to_end_correct_pct']}%")
    rt = s.get("retrieval", {})
    if rt:
        at = rt["attribution"]
        print(f"RETRIEVAL gold-in-topK: {rt['retrieval_hit_pct']:>5}%   "
              f"lost to -> retrieval_miss:{at['retrieval_miss']}  "
              f"refused_on_hit:{at['refused_on_hit']}  "
              f"answered_wrong:{at['answered_wrong']}  correct:{at['correct']}")
    print(f"COST      {s['cost']['wall_seconds']}s total, "
          f"{s['cost']['sec_per_q']}s/q")
    print("=" * 68)
    print(f"written {out_path}")
    print("A/B: run again with --max-steps 0 (control) and diff correct-refusal "
          "+ end-to-end-correct. That delta = CRAG's measured effect.")


def main():
    ap = argparse.ArgumentParser()
    crag.add_common_args(ap)
    ap.add_argument("--qa", default=str(crag.S5_DIR / "qa_stage5_v3.json"))
    ap.add_argument("--out", default="crag_eval.json")
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    args = ap.parse_args()

    app, parents = crag.build_from_args(args)
    app._crag_max_steps = args.max_steps  # stash for the report

    judge_llm = crag.make_chat_anthropic(
        args.judge_model, temperature=0).with_structured_output(Judgement)

    evaluate(app, parents, judge_llm, args.qa, args.out)


if __name__ == "__main__":
    main()
