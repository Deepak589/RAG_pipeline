#!/usr/bin/env python3
"""Stage 6 — build a gold-anchored N% corpus subset as its OWN Postgres database.

Why a separate DATABASE (not a collection in the same DB, not a metadata filter):
HNSW + GIN indexes are per-TABLE, and langchain-postgres uses one fixed table
(`langchain_pg_embedding`) for every collection. A 40% collection copied into the
same table would share the full corpus' HNSW graph; the ANN scan returns the
top ef_search rows of the WHOLE table and `collection_id` is applied AFTER —
selective post-filtering silently truncates results (same failure shape as the
ef_search bug). A separate DB = same table names, own indexes, zero code change
downstream: just point `--conn` + `--collection` at it.

What it does (no re-embedding, no re-labeling):
  1. sample parents: every gold parent in the QA file + random distractors up to N%
  2. write subset sections/chunks json + manifest (seed, counts) to subset_<tag>/
  3. create the new DB, copy the child rows (embeddings included) via binary COPY
  4. build HNSW + tsvector/GIN on the subset, ANALYZE
  5. verify: row/parent counts, gold coverage, HNSW self-recall @ef_search>=k

Run (on the Mac, from RAG/rag_stage_6):
    export PG_CONN="postgresql+psycopg://postgres:postgres@localhost:5432/ragdev"
    python build_subset.py --dry-run          # sampling + json only, no DB
    python build_subset.py                    # full build, 40%, seed 42
    python build_subset.py --reset            # drop + rebuild the subset DB
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
S5 = HERE.parent / "rag_stage_5"
DEFAULT_CONN = os.environ.get(
    "PG_CONN", "postgresql+psycopg://postgres:postgres@localhost:5432/ragdev")

EMBED_DIM = 1024          # BGE-M3
VERIFY_SAMPLES = 200      # rows used for the HNSW self-recall check


# ------------------------------------------------------------------ sampling

def gold_parents(qa_path):
    qs = json.loads(Path(qa_path).read_text())["questions"]
    return {p for q in qs for p in q.get("relevant_parent_ids", [])}


def sample_parents(all_ids, gold, frac, seed):
    """Gold-anchored sample: all gold parents + random distractors up to frac*N."""
    all_ids = sorted(all_ids)                      # sorted -> reproducible per seed
    missing = gold - set(all_ids)
    if missing:
        sys.exit(f"{len(missing)} gold parents not in sections.json, e.g. "
                 f"{sorted(missing)[:3]} — wrong sections file?")
    target = round(frac * len(all_ids))
    if target < len(gold):
        sys.exit(f"frac too small: target {target} < {len(gold)} gold parents")
    rest = [p for p in all_ids if p not in gold]
    fill = random.Random(seed).sample(rest, target - len(gold))
    return gold | set(fill)


def source_mix(ids):
    return Counter(p.split("/", 1)[0] for p in ids)


def write_subset_files(out_dir, keep, sections, chunks, meta):
    out_dir.mkdir(exist_ok=True)
    secs = [s for s in sections if f"{s['source']}#{s['section_idx']}" in keep]
    chs = [c for c in chunks if c["parent_id"] in keep]
    (out_dir / "sections.json").write_text(json.dumps(secs, ensure_ascii=False))
    (out_dir / "chunks.json").write_text(json.dumps(chs, ensure_ascii=False))
    meta = {**meta, "parents": len(secs), "children": len(chs)}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=1))
    return chs, meta


# ------------------------------------------------------------------ database

def dsn(conn):                       # SQLAlchemy URL -> libpq DSN
    return conn.replace("postgresql+psycopg://", "postgresql://")


def swap_db(conn, db):
    return conn.rsplit("/", 1)[0] + "/" + db


def create_db(src_conn, db, reset):
    import psycopg
    with psycopg.connect(dsn(src_conn), autocommit=True) as c:
        exists = c.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db,)).fetchone()
        if exists and not reset:
            sys.exit(f"database '{db}' exists — re-run with --reset to rebuild it")
        if exists:
            c.execute(f'DROP DATABASE "{db}" WITH (FORCE)')
        c.execute(f'CREATE DATABASE "{db}"')


def init_schema(dst_conn, collection):
    """Let langchain-postgres create tables + collection row (no hand-written DDL
    to drift). The fake embedder is never called — we copy real vectors."""
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_postgres import PGVector
    PGVector(embeddings=DeterministicFakeEmbedding(size=EMBED_DIM), connection=dst_conn,
             collection_name=collection, use_jsonb=True)


def collection_uuid(c, name):
    return c.execute("SELECT uuid FROM langchain_pg_collection WHERE name=%s",
                     (name,)).fetchone()[0]


def copy_rows(src_conn, dst_conn, src_col, dst_col, parent_ids):
    """Stream child rows for the kept parents src -> dst with binary COPY."""
    import psycopg
    with psycopg.connect(dsn(src_conn)) as src, psycopg.connect(dsn(dst_conn)) as dst:
        dst_uuid = collection_uuid(dst, dst_col)
        out_sql = ("COPY (SELECT e.id, %s::uuid, e.embedding, e.document, e.cmetadata "
                   "FROM langchain_pg_embedding e "
                   "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
                   "WHERE c.name = %s AND e.cmetadata->>'parent_id' = ANY(%s)) "
                   "TO STDOUT (FORMAT BINARY)")
        in_sql = ("COPY langchain_pg_embedding "
                  "(id, collection_id, embedding, document, cmetadata) "
                  "FROM STDIN (FORMAT BINARY)")
        t0, nbytes = time.time(), 0
        with src.cursor() as so, dst.cursor() as di:
            with so.copy(out_sql, (str(dst_uuid), src_col, sorted(parent_ids))) as co, \
                 di.copy(in_sql) as ci:
                for blob in co:
                    ci.write(blob)
                    nbytes += len(blob)
                    if nbytes // 2**26 != (nbytes - len(blob)) // 2**26:   # every 64MB
                        print(f"  copied {nbytes/2**20:.0f} MB  ({time.time()-t0:.0f}s)")
        dst.commit()
    print(f"copy done: {nbytes/2**20:.0f} MB in {time.time()-t0:.0f}s")


def verify(dst_conn, dst_col, keep, gold, n_children, ef_search):
    import psycopg
    with psycopg.connect(dsn(dst_conn), autocommit=True) as c:
        q = lambda sql, *a: c.execute(sql, a).fetchone()[0]
        rows = q("SELECT count(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection "
                 "col ON col.uuid=e.collection_id WHERE col.name=%s", dst_col)
        parents = q("SELECT count(DISTINCT cmetadata->>'parent_id') "
                    "FROM langchain_pg_embedding")
        got_gold = q("SELECT count(DISTINCT cmetadata->>'parent_id') FROM langchain_pg_embedding "
                     "WHERE cmetadata->>'parent_id' = ANY(%s)", sorted(gold))
        dim = q("SELECT vector_dims(embedding) FROM langchain_pg_embedding LIMIT 1")
        # HNSW self-recall: each sampled row, queried by its own vector, must come back at
        # distance ~0. Not "id == rank 1": the corpus has exact-duplicate chunk vectors, and
        # a tie at distance 0 may return the twin row instead.
        c.execute(f"SET hnsw.ef_search = {ef_search}")
        ids = [r[0] for r in c.execute(
            "SELECT id FROM langchain_pg_embedding ORDER BY random() LIMIT %s", (VERIFY_SAMPLES,))]
        ok = sum(c.execute(
            "SELECT embedding <=> q.v FROM langchain_pg_embedding, "
            "(SELECT embedding v FROM langchain_pg_embedding WHERE id=%s) q "
            "ORDER BY embedding <=> q.v LIMIT 1", (i,)
        ).fetchone()[0] < 1e-6 for i in ids)
    checks = {
        "rows == chunks.json": (rows, n_children),
        "distinct parents == kept": (parents, len(keep)),
        "gold parents present": (got_gold, len(gold)),
        "embedding dim": (dim, EMBED_DIM),
        f"HNSW self-recall @ef={ef_search}": (ok, len(ids)),
    }
    bad = False
    for name, (got, want) in checks.items():
        flag = "ok " if got == want else "BAD"
        bad |= got != want
        print(f"  [{flag}] {name}: {got} (expected {want})")
    if bad:
        sys.exit("verification FAILED — do not run evals on this DB")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qa", default=str(S5 / "qa_stage5_v3.json"))
    ap.add_argument("--sections", default=str(S5 / "sections.json"))
    ap.add_argument("--chunks", default=str(S5 / "chunks.json"))
    ap.add_argument("--src-conn", default=DEFAULT_CONN)
    ap.add_argument("--src-collection", default="rag_stage5__bge_m3")
    ap.add_argument("--ef-search", type=int, default=400)
    ap.add_argument("--reset", action="store_true", help="drop + rebuild the subset DB")
    ap.add_argument("--dry-run", action="store_true", help="sample + json only, no DB")
    args = ap.parse_args()

    tag = f"{round(args.frac * 100)}pct"
    dst_db = f"{args.src_conn.rsplit('/', 1)[1].split('?')[0]}_{tag}"
    dst_conn = swap_db(args.src_conn, dst_db)
    dst_col = f"{args.src_collection}__{tag}"

    print(f"loading sections/chunks ...")
    sections = json.loads(Path(args.sections).read_text())
    chunks = json.loads(Path(args.chunks).read_text())
    all_ids = [f"{s['source']}#{s['section_idx']}" for s in sections]
    gold = gold_parents(args.qa)
    keep = sample_parents(all_ids, gold, args.frac, args.seed)

    meta = {"corpus": tag, "frac": args.frac, "seed": args.seed, "db": dst_db,
            "collection": dst_col, "gold_parents": len(gold),
            "total_parents": len(all_ids), "qa": Path(args.qa).name}
    out_dir = HERE / f"subset_{tag}"
    chs, meta = write_subset_files(out_dir, keep, sections, chunks, meta)
    print(f"subset: {meta['parents']}/{len(all_ids)} parents "
          f"({len(gold)} gold + {meta['parents'] - len(gold)} distractors), "
          f"{meta['children']} children -> {out_dir.name}/")
    full, kept = source_mix(all_ids), source_mix(keep)
    print("  source mix kept/full: " + "  ".join(
        f"{s}:{kept[s]}/{full[s]}" for s in sorted(full)))

    if args.dry_run:
        return

    print(f"\n== DB '{dst_db}' ==")
    create_db(args.src_conn, dst_db, args.reset)
    init_schema(dst_conn, dst_col)
    copy_rows(args.src_conn, dst_conn, args.src_collection, dst_col, keep)
    sys.path.insert(0, str(S5))
    import ingest                      # lazy: pulls langchain deps, not needed for --dry-run
    ingest.ensure_indexes(dst_conn)
    import psycopg
    with psycopg.connect(dsn(dst_conn), autocommit=True) as c:
        c.execute("ANALYZE langchain_pg_embedding")
    print("\n== VERIFY ==")
    verify(dst_conn, dst_col, keep, gold, len(chs), args.ef_search)

    print(f"\nready. corpus={tag}. Eval with:\n"
          f"  python crag_eval.py --conn {dst_conn} --collection {dst_col} \\\n"
          f"      --sections {out_dir.name}/sections.json --chunks {out_dir.name}/chunks.json \\\n"
          f"      --out crag_eval_{tag}_<run>.json")


if __name__ == "__main__":
    main()
