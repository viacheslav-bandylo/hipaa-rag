from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Document
from app.services.embeddings import embedding_service
from app.services.bm25_service import bm25_service
from app.config import settings


class RetrievalService:
    # RRF constant (commonly 60)
    RRF_K = 60

    def __init__(self, session: AsyncSession):
        self.session = session

    async def search(self, query: str, top_k: int = None) -> list[Document]:
        """Vector-only search (legacy method)."""
        if top_k is None:
            top_k = settings.retrieval_top_k

        query_embedding = embedding_service.embed_text(query)

        stmt = text("""
            SELECT id, content, section_reference, part_number, section_number,
                   paragraph_reference, page_number, parent_context,
                   embedding <=> :query_embedding AS distance
            FROM documents
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
        """)

        result = await self.session.execute(
            stmt,
            {"query_embedding": str(query_embedding), "top_k": top_k}
        )

        rows = result.fetchall()

        documents = []
        for row in rows:
            doc = Document(
                id=row.id,
                content=row.content,
                section_reference=row.section_reference,
                part_number=row.part_number,
                section_number=row.section_number,
                paragraph_reference=row.paragraph_reference,
                page_number=row.page_number,
                parent_context=row.parent_context
            )
            documents.append(doc)

        return documents

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[int, int]]:
        """Internal vector search returning (doc_id, rank) pairs."""
        query_embedding = embedding_service.embed_text(query)

        stmt = text("""
            SELECT id
            FROM documents
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
        """)

        result = await self.session.execute(
            stmt,
            {"query_embedding": str(query_embedding), "top_k": top_k}
        )

        rows = result.fetchall()
        return [(row.id, rank + 1) for rank, row in enumerate(rows)]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, int]]:
        """Internal BM25 search returning (doc_id, rank) pairs."""
        results = bm25_service.search(query, top_k)
        return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(results)]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[int, int]],
        bm25_results: list[tuple[int, int]],
        k: int = None
    ) -> list[int]:
        """Combine vector and BM25 results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank)) for each ranking system

        Args:
            vector_results: List of (doc_id, rank) from vector search
            bm25_results: List of (doc_id, rank) from BM25 search
            k: RRF constant (default 60)

        Returns:
            List of document IDs sorted by fused score
        """
        if k is None:
            k = self.RRF_K

        rrf_scores: dict[int, float] = {}

        for doc_id, rank in vector_results:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        for doc_id, rank in bm25_results:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]

    async def _fetch_documents_by_ids(self, doc_ids: list[int]) -> list[Document]:
        """Fetch documents by IDs, preserving order."""
        if not doc_ids:
            return []

        stmt = text("""
            SELECT id, content, section_reference, part_number, section_number,
                   paragraph_reference, page_number, parent_context
            FROM documents
            WHERE id = ANY(:ids)
        """)

        result = await self.session.execute(stmt, {"ids": doc_ids})
        rows = result.fetchall()

        doc_map = {}
        for row in rows:
            doc = Document(
                id=row.id,
                content=row.content,
                section_reference=row.section_reference,
                part_number=row.part_number,
                section_number=row.section_number,
                paragraph_reference=row.paragraph_reference,
                page_number=row.page_number,
                parent_context=row.parent_context
            )
            doc_map[row.id] = doc

        return [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]

    async def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        use_bm25: bool = True
    ) -> list[Document]:
        """Hybrid search combining vector similarity and BM25 with RRF.

        Args:
            query: Search query
            top_k: Number of results to return
            use_bm25: Whether to use BM25 (fallback to vector-only if False or not indexed)

        Returns:
            List of Document objects sorted by fused relevance
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        candidate_pool_size = top_k * 3

        vector_results = await self._vector_search(query, candidate_pool_size)

        if not use_bm25 or not bm25_service.is_indexed:
            doc_ids = [doc_id for doc_id, _ in vector_results[:top_k]]
            return await self._fetch_documents_by_ids(doc_ids)

        bm25_results = self._bm25_search(query, candidate_pool_size)

        fused_doc_ids = self._reciprocal_rank_fusion(vector_results, bm25_results)

        return await self._fetch_documents_by_ids(fused_doc_ids[:top_k])

    async def get_by_section(self, section_reference: str) -> list[Document]:
        stmt = select(Document).where(
            Document.section_reference.ilike(f"%{section_reference}%")
        ).order_by(Document.page_number)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_document_count(self) -> int:
        result = await self.session.execute(text("SELECT COUNT(*) FROM documents"))
        return result.scalar()
