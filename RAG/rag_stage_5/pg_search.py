"""Postgres full-text search retriever — the sparse half of hybrid retrieval
for the pgvector-backed path.

Replaces the in-memory `BM25Retriever.from_documents(children)` (which loads
and tokenizes all 497K children on every process start — see
rag_stage_6/stage6_learning.md) with a query against the tsvector+GIN index
built by `ingest.py --index`. Ranking is `ts_rank_cd` (TF/proximity-based),
not true BM25 — a deliberate tradeoff, see stage6_learning.md.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class PGFullTextRetriever(BaseRetriever):
    """Full-text search over langchain_pg_embedding.document_tsv (GIN index)."""

    conn: str
    collection: str
    k: int = 200

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None
                                ) -> list[Document]:
        import psycopg

        dsn = self.conn.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as c, c.cursor() as cur:
            cur.execute(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection col ON col.uuid = e.collection_id
                WHERE col.name = %s
                  AND e.document_tsv @@ plainto_tsquery('simple', %s)
                ORDER BY ts_rank_cd(e.document_tsv, plainto_tsquery('simple', %s)) DESC
                LIMIT %s
                """,
                (self.collection, query, query, self.k),
            )
            rows = cur.fetchall()
        return [Document(page_content=doc, metadata=meta) for doc, meta in rows]
