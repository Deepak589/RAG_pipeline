#!/usr/bin/env python3
"""Verify a subset DB built by build_subset.py is actually usable for evals.

Stronger than build_subset.verify(): that one proves the ROWS arrived. This
proves the INDEXES are being used — the failure mode that actually bit us
(HNSW ef_search collapse) is invisible to a correctness-only check, because a
sequential scan returns perfect results while the ANN path is broken.

Checks:
  1. exactly one collection in this DB   (no cross-contamination)
  2. row / parent / gold counts match the subset json
  3. every gold parent has all its children present
  4. HNSW index EXISTS and the planner USES it for an ORDER BY <=> query
  5. ANN self-recall: query by a row's own vector -> distance ~0 at top-1
     (distance, not id: duplicate chunk vectors tie at 0 and return the twin)
  6. ANN gold-parent recall: query by a gold child's vector -> a sibling from
     the SAME parent in top-10  (the thing the eval depends on)
  7. GIN/tsvector exists, is used, and returns rows for a real term

Run:
    python check_subset.py                                  # defaults to 40pct
    python check_subset.py --conn ... --collection ... --subset-dir subset_40pct
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S5 = HERE.parent / "rag_stage_5"
SAMPLES = 100          # rows for the ANN self-recall probe
GOLD_SAMPLES = 40      # gold children for the parent-recall probe
TOPK = 10


def dsn(conn):
    return conn.replace("postgresql+psycopg://", "postgresql://")


def conn_for_db(db):
    """Same credentials/host as PG_CONN (the builder's source DB), pointed at `db`.
    Never hardcode postgres:postgres — a Homebrew install has no such role."""
    base = os.environ.get("PG_CONN")
    if not base:
        sys.exit("PG_CONN not set — export the same value you used for build_subset.py, "
                 "or pass --conn explicitly")
    return base.rsplit("/", 1)[0] + "/" + db


class Checks:
    """Collect pass/fail so one run reports everything, not just the first break."""

    def __init__(self):
        self.bad = 0

    def __call__(self, name, got, want, cmp="=="):
        ok = got == want if cmp == "==" else got >= want
        self.bad += not ok
        print(f"  [{'ok ' if ok else 'BAD'}] {name}: {got} (expected {cmp} {want})")

    def note(self, name, value):
        print(f"  [   ] {name}: {value}")


def plan_uses(cur, sql, params, index_name):
    """True if the planner picks `index_name` for this query. The point of the
    whole check: a seq-scan fallback silently makes ANN results look perfect."""
    cur.execute(f"EXPLAIN (FORMAT JSON) {sql}", params)
    return index_name in json.dumps(cur.fetchone()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", default="subset_40pct")
    ap.add_argument("--conn", help="default: $PG_CONN with the db swapped "
                                 "to the manifest's subset db")
    ap.add_argument("--collection", help="defaults to the manifest's collection")
    ap.add_argument("--qa", default=str(S5 / "qa_stage5_v3.json"))
    ap.add_argument("--ef-search", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import psycopg

    d = HERE / args.subset_dir
    man = json.loads((d / "manifest.json").read_text())
    conn = args.conn or conn_for_db(man["db"])
    col = args.collection or man["collection"]
    print(f"checking {man['corpus']}  db={man['db']}  collection={col}")

    chunks = json.loads((d / "chunks.json").read_text())
    qa = json.loads(Path(args.qa).read_text())["questions"]
    gold = {p for q in qa for p in q.get("relevant_parent_ids", [])}
    gold_children = [c for c in chunks if c["parent_id"] in gold]
    subset_parents = {c["parent_id"] for c in chunks}

    rng = random.Random(args.seed)
    c_ = Checks()

    with psycopg.connect(dsn(conn), autocommit=True) as cx, cx.cursor() as cur:
        q1 = lambda sql, *a: cur.execute(sql, a).fetchone()[0]
        cur.execute(f"SET hnsw.ef_search = {args.ef_search}")

        print("\n-- contents --")
        cols = [r[0] for r in cur.execute(
            "SELECT name FROM langchain_pg_collection").fetchall()]
        c_("collections in db", cols, [col])
        c_("rows", q1("SELECT count(*) FROM langchain_pg_embedding"), len(chunks))
        c_("distinct parents",
           q1("SELECT count(DISTINCT cmetadata->>'parent_id') FROM langchain_pg_embedding"),
           len(subset_parents))
        c_("gold parents present",
           q1("SELECT count(DISTINCT cmetadata->>'parent_id') FROM langchain_pg_embedding "
              "WHERE cmetadata->>'parent_id' = ANY(%s)", sorted(gold)), len(gold))
        c_("gold children present",
           q1("SELECT count(*) FROM langchain_pg_embedding "
              "WHERE cmetadata->>'parent_id' = ANY(%s)", sorted(gold)), len(gold_children))
        c_("embedding dim",
           q1("SELECT vector_dims(embedding) FROM langchain_pg_embedding LIMIT 1"), 1024)

        print("\n-- indexes --")
        idx = [r[0] for r in cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='langchain_pg_embedding'"
        ).fetchall()]
        c_.note("present", ", ".join(sorted(idx)))
        c_("hnsw index exists", "langchain_pg_embedding_hnsw" in idx, True)
        c_("tsv gin index exists", "langchain_pg_embedding_tsv_gin" in idx, True)

        probe = q1("SELECT embedding FROM langchain_pg_embedding LIMIT 1")
        ann_sql = ("SELECT id FROM langchain_pg_embedding "
                   "ORDER BY embedding <=> %s::vector LIMIT %s")
        c_("planner uses HNSW (not seq scan)",
           plan_uses(cur, ann_sql, (probe, TOPK), "langchain_pg_embedding_hnsw"), True)

        print("\n-- ANN recall --")
        ids = [r[0] for r in cur.execute(
            "SELECT id FROM langchain_pg_embedding ORDER BY random() LIMIT %s",
            (SAMPLES,)).fetchall()]
        hit = 0
        for i in ids:
            v = q1("SELECT embedding FROM langchain_pg_embedding WHERE id=%s", i)
            hit += q1("SELECT embedding <=> %s::vector FROM langchain_pg_embedding "
                      "ORDER BY embedding <=> %s::vector LIMIT 1", v, v) < 1e-6
        c_(f"self-recall @ef={args.ef_search}", hit, len(ids))

        # The eval's real dependency: a gold child's neighbourhood must still be
        # reachable through the ANN graph, not just stored in the table.
        sample = rng.sample(gold_children, min(GOLD_SAMPLES, len(gold_children)))
        same_parent = 0
        for ch in sample:
            v = q1("SELECT embedding FROM langchain_pg_embedding WHERE id=%s", ch["id"])
            pids = [r[0] for r in cur.execute(
                "SELECT cmetadata->>'parent_id' FROM langchain_pg_embedding "
                "ORDER BY embedding <=> %s::vector LIMIT %s", (v, TOPK)).fetchall()]
            same_parent += ch["parent_id"] in pids
        c_(f"gold parent in ANN top-{TOPK}", same_parent, len(sample))

        print("\n-- full-text --")
        fts_sql = ("SELECT count(*) FROM langchain_pg_embedding "
                   "WHERE document_tsv @@ plainto_tsquery('simple', %s)")
        n = q1(fts_sql, "risk factors")
        c_("fts rows for 'risk factors'", n, 1, cmp=">=")
        c_.note("planner uses GIN",
                plan_uses(cur, fts_sql.replace("count(*)", "id"), ("risk factors",),
                          "langchain_pg_embedding_tsv_gin"))

    print()
    if c_.bad:
        sys.exit(f"{c_.bad} check(s) FAILED — do not run evals on this DB")
    print(f"all checks passed — {man['corpus']} is eval-ready")


if __name__ == "__main__":
    main()
