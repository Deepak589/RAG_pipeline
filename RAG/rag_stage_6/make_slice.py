#!/usr/bin/env python3
"""Cut a reproducible eval slice from qa_stage5_v3.json.

Output is a drop-in QA file (same {version, questions} schema), so it runs with
`crag_eval.py --qa <slice>` unchanged. The version string encodes mode/size/seed
so a result can never be mistaken for a full-160 run.

Two modes, for two different jobs:

  balanced  — proportional by question type. An unbiased mini-eval; use for a
              knob that moves EVERYTHING (retrieval: B1 blurbs, B2 tsquery).

  cause     — every question that landed in the named baseline failure cause,
              plus all negatives, plus a control sample of currently-correct
              ones. Use for a knob aimed at ONE failure class (A1, A2).
              A balanced slice is the wrong instrument there: at 30 Q it carries
              ~3 of the 13 refused_on_hit cases, far too few to see the effect,
              while the cause slice carries all 13 for the same wall time.

  Negatives are included in full by default in `cause` mode: A1/A2 loosen the
  refusal path, and correct-refusal (currently 100%) is exactly what they put
  at risk. Measuring a loosening knob without the population it endangers is
  how a regression ships.

Run:
    python make_slice.py --mode cause --cause refused_on_hit \
        --baseline crag_grader_v2.json --out qa_slice_a1.json
    python make_slice.py --mode balanced -n 30 --out qa_slice30.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
S5 = HERE.parent / "rag_stage_5"


def load_qa(path):
    d = json.loads(Path(path).read_text())
    return d["version"], d["questions"]


def causes_from(baseline):
    """id -> baseline attribution cause (positives only; negatives have none)."""
    rows = json.loads(Path(baseline).read_text())["rows"]
    return {r["id"]: r["cause"] for r in rows if "cause" in r}


def balanced(qs, n, rng):
    """Proportional by type, largest-remainder so the counts sum to exactly n."""
    by = {}
    for q in qs:
        by.setdefault(q["type"], []).append(q)
    exact = {t: len(v) * n / len(qs) for t, v in by.items()}
    take = {t: int(x) for t, x in exact.items()}
    for t in sorted(by, key=lambda t: exact[t] - take[t], reverse=True):
        if sum(take.values()) >= n:
            break
        take[t] += 1
    out = []
    for t, v in by.items():
        out += rng.sample(v, min(take[t], len(v)))
    return out


def by_cause(qs, cause_of, causes, n_control, all_negatives, rng):
    targeted = [q for q in qs if cause_of.get(q["id"]) in causes]
    negs = [q for q in qs if q["type"] == "negative"]
    chosen = {q["id"] for q in targeted}
    out = list(targeted)
    if all_negatives:
        out += [q for q in negs if q["id"] not in chosen]
        chosen |= {q["id"] for q in negs}
    controls = [q for q in qs if cause_of.get(q["id"]) == "correct"
                and q["id"] not in chosen]
    out += rng.sample(controls, min(n_control, len(controls)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa", default=str(S5 / "qa_stage5_v3.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["balanced", "cause"], default="cause")
    ap.add_argument("-n", type=int, default=30, help="balanced mode: slice size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="crag_grader_v2.json",
                    help="cause mode: eval json whose attribution defines the strata")
    ap.add_argument("--cause", nargs="+", default=["refused_on_hit"],
                    choices=["refused_on_hit", "retrieval_miss", "answered_wrong",
                             "correct"])
    ap.add_argument("--n-control", type=int, default=15,
                    help="cause mode: currently-correct questions kept as a "
                         "regression tripwire")
    ap.add_argument("--no-negatives", action="store_true",
                    help="cause mode: drop the negatives (NOT advised for A1/A2)")
    args = ap.parse_args()

    ver, qs = load_qa(args.qa)
    rng = random.Random(args.seed)

    if args.mode == "balanced":
        picked = balanced(qs, args.n, rng)
        tag = f"{ver}-slice{len(picked)}-balanced-s{args.seed}"
    else:
        cause_of = causes_from(args.baseline)
        picked = by_cause(qs, cause_of, set(args.cause), args.n_control,
                          not args.no_negatives, rng)
        tag = (f"{ver}-slice{len(picked)}-cause[{'+'.join(sorted(args.cause))}]"
               f"-s{args.seed}")

    order = {q["id"]: i for i, q in enumerate(qs)}       # keep original order
    picked.sort(key=lambda q: order[q["id"]])
    Path(args.out).write_text(json.dumps(
        {"version": tag, "questions": picked}, ensure_ascii=False, indent=1))

    print(f"{args.out}: {len(picked)} questions   version={tag}")
    print("  by type :", dict(Counter(q["type"] for q in picked)))
    if args.mode == "cause":
        cause_of = causes_from(args.baseline)
        print("  by cause:", dict(Counter(
            cause_of.get(q["id"], "negative") for q in picked)))
    print(f"  est. cost: ~{len(picked) * 8 // 60}m at 8.0 s/q "
          f"(full 160 = ~21m)")


if __name__ == "__main__":
    main()
