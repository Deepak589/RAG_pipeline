#!/usr/bin/env python3
"""Stage 5 — parser-routed ingestion: mixed corpus -> canonical chunks -> pgvector.

RUN ON YOUR MAC. Reads RAG/corpus/ (built by ingest/corpus_puller.py), routes
each document to the RIGHT loader by type, parent/child-chunks it, writes the
project's canonical `sections.json` (parents) + `chunks.json` (children) — so
run_eval.py / qa_gen.py / lc_pipeline.py all keep working unchanged — AND
upserts the child embeddings into a Postgres+pgvector collection for a real,
persistent, scalable dense index.

"Routing" here = FILE-TYPE routing (the thing you meant), not agentic query
routing. Three lanes, matched to what's actually in the corpus:

    digital PDF  (arxiv, wikipedia, rtd, rag_papers) -> PyMuPDFLoader   (fast)
    HTML         (sec *.htm)                          -> BSHTMLLoader
    scanned PDF  (archive, ~21MB each)               -> DoclingLoader + OCR (SLOW)
    markdown     (*.md)                               -> TextLoader

Why two chunk levels: small children embed to sharp vectors (good retrieval),
whole parents give the generator context — the same parent-child idea from
stage 2, now applied to every source.

Resumable: parsing checkpoints per document; a re-run skips doc_ids already in
chunks.json (so the slow OCR pass can run overnight and survive a Ctrl-C). The
embed pass is a separate, faster phase you can redo with --reset.

PARSING QUALITY IS A SILENT KNOB. A parser that returns empty/garbage text (a
failed OCR, a mangled table) reads downstream as a retrieval miss, not a parse
bug. This script logs a WARNING for any doc whose extracted text is suspiciously
short — eyeball those, especially the scanned lane, before trusting eval numbers.

Deps:
    pip install langchain langchain-community langchain-postgres langchain-huggingface \
                langchain-text-splitters pymupdf beautifulsoup4 lxml psycopg[binary] \
                sentence-transformers
    pip install langchain-docling docling            # scanned/OCR lane only
    # Postgres with pgvector, e.g.:
    #   docker run -d --name pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16

Run:
    export PG_CONN="postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    python ingest.py --parse                # route + chunk -> sections.json/chunks.json
    python ingest.py --embed                # chunks.json  -> pgvector
    python ingest.py --index                # HNSW + full-text indexes (one-time, idempotent)
    python ingest.py --all                  # parse + embed + index (default)
    python ingest.py --parse --only archive # just the scanned lane
    python ingest.py --embed --reset        # rebuild the pgvector collection

--index builds a vector HNSW index (dense ANN) and a tsvector+GIN index
(sparse full-text) directly on langchain_pg_embedding. At 497K rows this is
a one-time, multi-minute build (HNSW graph construction + a table rewrite
for the generated tsvector column) — expected, not a bug. Safe to re-run
(IF NOT EXISTS); only needed again if the collection is rebuilt (--reset).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).resolve().parent            # RAG/rag_stage_5
CORPUS_DIR = HERE.parent / "corpus"
SECTIONS_OUT = HERE / "sections.json"
CHUNKS_OUT = HERE / "chunks.json"
FAILED_OUT = HERE / "parse_failures.json"   # doc_id -> reason; skipped on re-run

PG_CONN = os.environ.get(
    "PG_CONN", "postgresql+psycopg://postgres:postgres@localhost:5432/ragdev")
# Dense embedder. Swapped all-MiniLM-L6-v2 (384-dim) -> BGE-M3 (1024-dim).
# BGE-M3: 8192-token context (no more silent 256-token truncation), retrieval-
# instruction-free (NO query prefix needed, unlike E5 / bge-*-v1.5), and the
# same model can later serve learned-sparse + ColBERT — activated one knob at a
# time, NOT here. ~2.3GB download; slow on CPU (use GPU / batch for the 70k pass).
EMBED_MODEL = "BAAI/bge-m3"
EMBED_NORMALIZE = True        # BGE expects L2-normalized vectors for cosine

# Collection name is tagged by model so the new 1024-dim vectors can't collide
# with the old 384-dim MiniLM index. Switching model => new collection => the
# first embed run needs --reset (dim change; PGVector can't mix dims in one).
def _model_tag(name: str) -> str:
    return name.split("/")[-1].replace(".", "_").replace("-", "_").lower()

COLLECTION = f"rag_stage5__{_model_tag(EMBED_MODEL)}"

PARENT_CHARS, PARENT_OVERLAP = 2000, 200
CHILD_CHARS, CHILD_OVERLAP = 450, 80
MIN_DOC_CHARS = 200          # below this = likely a parse/OCR failure -> WARN
CHECKPOINT_EVERY = 25        # flush sections/chunks every N docs (resumable)
EMBED_BATCH = 500

# subdir -> lane. Anything not listed falls back to extension-based routing.
SOURCE_LANE = {
    "arxiv": "pdf", "wikipedia": "pdf", "rtd": "pdf", "rag_papers": "pdf",
    "sec": "html", "archive": "scanned", "datagov": "pdf",
}


# ------------------------------------------------------------- lane -> loader

def load_text(path: Path, lane: str) -> str:
    """Return the full extracted text for one document, routed by lane."""
    if lane == "pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        docs = PyMuPDFLoader(str(path)).load()
    elif lane == "html":
        from langchain_community.document_loaders import BSHTMLLoader
        docs = BSHTMLLoader(str(path), open_encoding="utf-8").load()
    elif lane == "md":
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(str(path), encoding="utf-8").load()
    elif lane == "scanned":
        # Docling auto-OCRs scanned PDFs. This is the one lane not runnable in
        # the Cowork sandbox — if the import/signature differs on your machine,
        # this is the only spot to adjust. `pip install langchain-docling docling`.
        try:
            from langchain_docling import DoclingLoader
        except ImportError:
            raise RuntimeError("scanned lane needs: pip install langchain-docling docling")
        docs = DoclingLoader(file_path=str(path)).load()
    else:
        raise ValueError(f"unknown lane {lane}")
    return "\n".join(d.page_content for d in docs if d.page_content)


def lane_for(path: Path) -> str:
    src = path.parent.name
    if src in SOURCE_LANE:
        lane = SOURCE_LANE[src]
    else:
        lane = "pdf"
    ext = path.suffix.lower()
    if ext in (".htm", ".html"):
        return "html"
    if ext == ".md":
        return "md"
    if ext == ".txt":
        return "md"
    return lane


# --------------------------------------------------------------- chunking

# SEC filings (10-K/10-Q) carry NO <h*> tags — probed a real Wells Fargo 10-K:
# zero h1/h2/h3, but 22 clean "ITEM 1A. RISK FACTORS"-style headings. That's the
# reliable structure signal for the html/sec lane. (arxiv PDFs have NO stable
# heading signal across LaTeX templates — font-size detection gave 0/2/5/42 on
# four papers — so those stay char-windowed until eval justifies Docling.)
ITEM_RE = re.compile(r"(?im)^[ \t]*(item[ \t]+\d+[a-z]?\.?)\s+([A-Z][^\n]{0,80})")


def section_units(text, lane):
    """Return [(title_or_None, body)] — the section-aware parent units for a doc.

    sec/html -> split on ITEM headings (each ITEM = one coherent section, titled).
    every other lane -> a single (None, whole-doc) unit, i.e. UNCHANGED behaviour
    (the parent char-splitter windows it downstream, exactly as before). So this
    is a ONE-KNOB change: only the SEC lane gets structure-aware parents."""
    if lane == "html":
        hits = list(ITEM_RE.finditer(text))
        if len(hits) >= 3:                      # looks like a real filing skeleton
            units = []
            if hits[0].start() > 400:           # cover page / front matter
                units.append(("Front Matter", text[:hits[0].start()]))
            for i, m in enumerate(hits):
                end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
                title = re.sub(r"\s+", " ", m.group(0)).strip()[:80]
                units.append((title, text[m.start():end]))
            return units
    return [(None, text)]


def chunk_document(doc_id, source, text, parent_splitter, child_splitter, lane="pdf"):
    """text -> (parent_records, child_records) in the project's canonical schema.

    parent_id = f"{source}#{section_idx}" so it matches load_parents() in
    run_eval.py / lc_pipeline.py. source == doc_id (unique per document).

    Section-aware: units come from section_units(); each unit is still char-window
    size-bounded by parent_splitter, so a giant ITEM (Risk Factors can be 20k+
    chars) is split into several parents that all KEEP the ITEM title — that's the
    stage-3 'split giant sections, keep their heading' lesson, applied to SEC."""
    parents, children = [], []
    pidx = 0
    for sec_title, body in section_units(text, lane):
        for ptext in parent_splitter.split_text(body):
            pid = f"{doc_id}#{pidx}"
            title = sec_title or f"{Path(doc_id).name} [{pidx}]"
            parents.append({
                "source": doc_id, "section_idx": pidx,
                "title": title, "text": ptext,
            })
            for cidx, ctext in enumerate(child_splitter.split_text(ptext)):
                children.append({
                    "id": f"{pid}#{cidx}", "parent_id": pid,
                    "source": doc_id, "text": ctext,
                })
            pidx += 1
    return parents, children


# ------------------------------------------------------------------ parse phase

def iter_corpus(only=None, limit=None):
    """Yield (doc_id, path) for every file under corpus/<source>/, skipping
    manifest/junk. doc_id = '<source>/<stem>' — stable and unique."""
    n = 0
    for sub in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir()):
        if only and sub.name != only:
            continue
        for f in sorted(sub.iterdir()):
            if f.name.startswith(".") or f.suffix.lower() not in (
                    ".pdf", ".htm", ".html", ".md", ".txt"):
                continue
            yield f"{sub.name}/{f.stem}", f
            n += 1
            if limit and n >= limit:
                return


def parse_phase(only, limit):
    parents_all = json.loads(SECTIONS_OUT.read_text()) if SECTIONS_OUT.exists() else []
    children_all = json.loads(CHUNKS_OUT.read_text()) if CHUNKS_OUT.exists() else []
    failed = json.loads(FAILED_OUT.read_text()) if FAILED_OUT.exists() else {}
    done = {c["source"] for c in children_all}          # resume: skip parsed docs
    done |= failed.keys()                                # and permanently-broken ones

    p_split = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHARS, chunk_overlap=PARENT_OVERLAP)
    c_split = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHARS, chunk_overlap=CHILD_OVERLAP)

    def flush():
        SECTIONS_OUT.write_text(json.dumps(parents_all, ensure_ascii=False, indent=2))
        CHUNKS_OUT.write_text(json.dumps(children_all, ensure_ascii=False, indent=2))

    processed = 0
    empties = []
    for doc_id, path in iter_corpus(only, limit):
        if doc_id in done:
            continue
        lane = lane_for(path)
        try:
            text = load_text(path, lane).strip()
        except Exception as e:
            print(f"  FAIL [{lane}] {doc_id}: {str(e)[:90]}")
            failed[doc_id] = str(e)[:200]
            FAILED_OUT.write_text(json.dumps(failed, ensure_ascii=False, indent=2))
            continue
        if len(text) < MIN_DOC_CHARS:
            empties.append(doc_id)
            print(f"  WARN empty/short ({len(text)}c) [{lane}] {doc_id} "
                  f"— probable parse/OCR failure, skipped")
            continue
        parents, children = chunk_document(doc_id, doc_id, text, p_split, c_split, lane)
        parents_all.extend(parents)
        children_all.extend(children)
        processed += 1
        print(f"  [{lane}] {doc_id}: {len(parents)}p/{len(children)}c "
              f"({len(text)//1000}k chars)")
        if processed % CHECKPOINT_EVERY == 0:
            flush()
            print(f"  ...checkpoint ({processed} new docs, "
                  f"{len(children_all)} children total)")
    flush()
    print(f"\nPARSE done: {processed} new docs, "
          f"{len(parents_all)} parents / {len(children_all)} children total")
    if failed:
        print(f"WARN: {len(failed)} docs permanently failed to parse (unloadable — "
              f"e.g. DRM/encrypted) — logged in {FAILED_OUT.name}, skipped on future runs")
    if empties:
        print(f"WARN: {len(empties)} docs produced no usable text "
              f"(check these parsers): {empties[:8]}{'...' if len(empties)>8 else ''}")


# ------------------------------------------------------------------ embed phase

def embed_phase(reset, conn, collection, limit=None):
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_postgres import PGVector

    if not CHUNKS_OUT.exists():
        sys.exit("chunks.json missing — run: python ingest.py --parse")
    children = json.loads(CHUNKS_OUT.read_text())
    if limit:
        children = children[:limit]
    print(f"embedding {len(children)} children into pgvector "
          f"collection '{collection}'  model={EMBED_MODEL}  (reset={reset})")

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE},
    )
    store = PGVector(embeddings=emb, connection=conn, collection_name=collection,
                     use_jsonb=True, pre_delete_collection=reset)

    for i in range(0, len(children), EMBED_BATCH):
        batch = children[i:i + EMBED_BATCH]
        docs = [Document(page_content=c["text"].replace("\x00", ""),
                         metadata={"parent_id": c["parent_id"],
                                   "child_id": c["id"], "source": c["source"]})
                for c in batch]
        store.add_documents(docs, ids=[c["id"] for c in batch])   # id = idempotent upsert
        print(f"  embedded {min(i+EMBED_BATCH, len(children))}/{len(children)}")
    print(f"pgvector collection '{collection}' ready on {conn.split('@')[-1]}")


# ------------------------------------------------------------------- index phase

def ensure_indexes(conn):
    """HNSW (dense ANN) + tsvector/GIN (sparse full-text) on langchain_pg_embedding.
    Idempotent — safe to re-run. Replaces brute-force vector scan and the
    in-memory rank_bm25 corpus (see rag_stage_6/stage6_learning.md)."""
    import psycopg
    dsn = conn.replace("postgresql+psycopg://", "postgresql://")
    print(f"building indexes on {conn.split('@')[-1]} (one-time, may take minutes)...")
    with psycopg.connect(dsn, autocommit=True) as c, c.cursor() as cur:
        # PGVector creates `embedding` as an untyped `vector` column (no fixed
        # dimension) — HNSW requires a fixed-dim column. All rows are already
        # 1024-dim (BGE-M3), so this is a safe typmod-only ALTER.
        cur.execute("""
            ALTER TABLE langchain_pg_embedding
              ALTER COLUMN embedding TYPE vector(1024)
        """)
        # 'simple' (no stemming/stopwords) matches the hand-tuned lowercase
        # tokenizer stage-5's rank_bm25 path used (lc_pipeline.bm25_tokenize) —
        # 'english' stemming was tried first and measurably hurt grading
        # recall (see rag_stage_6/stage6_learning.md A/B numbers).
        cur.execute("""
            ALTER TABLE langchain_pg_embedding
              ADD COLUMN IF NOT EXISTS document_tsv tsvector
              GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_tsv_gin
              ON langchain_pg_embedding USING gin (document_tsv)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw
              ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops)
        """)
    print("indexes ready: langchain_pg_embedding_hnsw, langchain_pg_embedding_tsv_gin")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true", help="route + chunk -> json")
    ap.add_argument("--embed", action="store_true", help="chunks.json -> pgvector")
    ap.add_argument("--all", action="store_true", help="parse, embed, index (default)")
    ap.add_argument("--index", action="store_true",
                    help="build HNSW + full-text indexes (idempotent)")
    ap.add_argument("--only", help="restrict to one corpus subdir (e.g. archive)")
    ap.add_argument("--limit", type=int, help="cap docs (quick trial)")
    ap.add_argument("--reset", action="store_true", help="drop+rebuild pgvector collection")
    ap.add_argument("--conn", default=PG_CONN)
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    any_explicit = args.parse or args.embed or args.index
    do_parse = args.parse or args.all or not any_explicit
    do_embed = args.embed or args.all or not any_explicit
    do_index = args.index or args.all or not any_explicit

    if not CORPUS_DIR.exists():
        sys.exit(f"corpus not found at {CORPUS_DIR} — run ingest/corpus_puller.py first")

    if do_parse:
        print("== PARSE ==")
        parse_phase(args.only, args.limit)
    if do_embed:
        print("\n== EMBED ==")
        embed_phase(args.reset, args.conn, args.collection, args.limit)
    if do_index:
        print("\n== INDEX ==")
        ensure_indexes(args.conn)


if __name__ == "__main__":
    main()
