"""Postgres full-text search retriever — the sparse half of hybrid retrieval
for the pgvector-backed path.

Replaces the in-memory `BM25Retriever.from_documents(children)` (which loads
and tokenizes all 497K children on every process start — see
rag_stage_6/stage6_learning.md) with a query against the tsvector+GIN index
built by `ingest.py --index`. Ranking is `ts_rank_cd` (TF/proximity-based),
not true BM25 — a deliberate tradeoff, see stage6_learning.md.
"""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# 'simple' tsvector config keeps stopwords, so they must be dropped here or an OR
# query matches nearly every row on "the"/"what".
_STOP = set("the a an of to in for on and or is are was were be been being this that "
            "these those with as by at from it its their his her what which who whom "
            "how why when where does do did can could would should will may might has "
            "have had not no than then into about over under between".split())


def or_tsquery(query: str) -> str:
    """'what was X revenue' -> 'x | revenue' (content words, OR-joined)."""
    return " | ".join(dict.fromkeys(
        t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOP))


class PGFullTextRetriever(BaseRetriever):
    """Full-text search over langchain_pg_embedding.document_tsv (GIN index).

    mode="plain" — plainto_tsquery: ANDs EVERY word (incl. stopwords under
                   'simple'), so a natural-language question almost never matches
                   (measured 0/141 positives on the 40pct corpus) -> sparse side dead.
    mode="or"    — OR over content words, ranked by ts_rank_cd.
    # ponytail: ts_rank_cd has no IDF, so common content words weigh like rare
    # ones; move to a real BM25 (pg_search/ParadeDB) if OR ranking stays noisy.
    """

    conn: str
    collection: str
    k: int = 200
    mode: str = "plain"

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None
                                ) -> list[Document]:
        import psycopg

        if self.mode == "or":
            fn, query = "to_tsquery", or_tsquery(query)
            if not query:
                return []
        else:
            fn = "plainto_tsquery"
        dsn = self.conn.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as c, c.cursor() as cur:
            cur.execute(
                f"""
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection col ON col.uuid = e.collection_id
                WHERE col.name = %s
                  AND e.document_tsv @@ {fn}('simple', %s)
                ORDER BY ts_rank_cd(e.document_tsv, {fn}('simple', %s)) DESC
                LIMIT %s
                """,
                (self.collection, query, query, self.k),
            )
            rows = cur.fetchall()
        return [Document(page_content=doc, metadata=meta) for doc, meta in rows]
