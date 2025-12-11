import logging
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Document
from app.services.embeddings import embedding_service
from app.services.bm25_service import bm25_service
from app.services.reranker import reranker_service
from app.config import settings


logger = logging.getLogger(__name__)


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

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        section_filters: Optional[list[str]] = None,
        part_filters: Optional[list[str]] = None
    ) -> list[tuple[int, int]]:
        """Internal vector search returning (doc_id, rank) pairs.

        Args:
            query: Search query
            top_k: Number of results
            section_filters: Optional list of section numbers to filter by
            part_filters: Optional list of part numbers to filter by

        Returns:
            List of (doc_id, rank) tuples
        """
        query_embedding = embedding_service.embed_text(query)

        # Build WHERE clause based on filters
        where_clauses = []
        params = {"query_embedding": str(query_embedding), "top_k": top_k}

        if section_filters:
            # Match any of the provided section numbers
            placeholders = [f":section_{i}" for i in range(len(section_filters))]
            where_clauses.append(f"section_number IN ({', '.join(placeholders)})")
            for i, section in enumerate(section_filters):
                params[f"section_{i}"] = section

        if part_filters and not section_filters:
            # Only apply part filter if no section filter (section is more specific)
            placeholders = [f":part_{i}" for i in range(len(part_filters))]
            where_clauses.append(f"part_number IN ({', '.join(placeholders)})")
            for i, part in enumerate(part_filters):
                params[f"part_{i}"] = part

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        stmt = text(f"""
            SELECT id
            FROM documents
            {where_sql}
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
        """)

        result = await self.session.execute(stmt, params)

        rows = result.fetchall()
        return [(row.id, rank + 1) for rank, row in enumerate(rows)]

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: Optional[set[int]] = None
    ) -> list[tuple[int, int]]:
        """Internal BM25 search returning (doc_id, rank) pairs.

        Args:
            query: Search query
            top_k: Number of results
            allowed_doc_ids: Optional set of doc IDs to filter results

        Returns:
            List of (doc_id, rank) tuples
        """
        # Get more results than needed if we're filtering
        fetch_k = top_k * 3 if allowed_doc_ids else top_k
        results = bm25_service.search(query, fetch_k)

        if allowed_doc_ids:
            # Filter to only allowed IDs and re-rank
            filtered = [(doc_id, score) for doc_id, score in results if doc_id in allowed_doc_ids]
            results = filtered[:top_k]

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

    async def _get_filtered_doc_ids(
        self,
        section_filters: Optional[list[str]] = None,
        part_filters: Optional[list[str]] = None
    ) -> Optional[set[int]]:
        """Get set of document IDs matching the filters.

        Used to pre-filter BM25 results which can't do SQL filtering.

        Returns:
            Set of matching doc IDs, or None if no filters applied
        """
        if not section_filters and not part_filters:
            return None

        where_clauses = []
        params = {}

        if section_filters:
            placeholders = [f":section_{i}" for i in range(len(section_filters))]
            where_clauses.append(f"section_number IN ({', '.join(placeholders)})")
            for i, section in enumerate(section_filters):
                params[f"section_{i}"] = section

        if part_filters and not section_filters:
            placeholders = [f":part_{i}" for i in range(len(part_filters))]
            where_clauses.append(f"part_number IN ({', '.join(placeholders)})")
            for i, part in enumerate(part_filters):
                params[f"part_{i}"] = part

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        stmt = text(f"SELECT id FROM documents {where_sql}")
        result = await self.session.execute(stmt, params)
        rows = result.fetchall()

        return {row.id for row in rows}

    async def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        use_bm25: bool = True,
        section_filters: Optional[list[str]] = None,
        part_filters: Optional[list[str]] = None,
        use_reranker: bool = True
    ) -> list[Document]:
        """Hybrid search combining vector similarity and BM25 with RRF.

        Args:
            query: Search query
            top_k: Number of results to return
            use_bm25: Whether to use BM25 (fallback to vector-only if False or not indexed)
            section_filters: Optional list of section numbers to filter by (e.g., ["164.502"])
            part_filters: Optional list of part numbers to filter by (e.g., ["164"])
            use_reranker: Whether to apply cross-encoder reranking

        Returns:
            List of Document objects sorted by fused relevance
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        # If reranker is enabled, over-retrieve to get more candidates for reranking
        should_rerank = use_reranker and settings.reranker_enabled
        if should_rerank:
            candidate_pool_size = settings.retrieval_candidates
        else:
            candidate_pool_size = top_k * 3

        # Log filter application
        if section_filters:
            logger.info(f"Applying section filter: {section_filters}")
        if part_filters:
            logger.info(f"Applying part filter: {part_filters}")

        # Get vector results with filters applied in SQL
        vector_results = await self._vector_search(
            query, candidate_pool_size, section_filters, part_filters
        )

        if not use_bm25 or not bm25_service.is_indexed:
            doc_ids = [doc_id for doc_id, _ in vector_results]
            documents = await self._fetch_documents_by_ids(doc_ids)
            if should_rerank:
                logger.info(f"Reranking {len(documents)} candidates")
                documents = reranker_service.rerank(query, documents, top_k)
            return documents[:top_k]

        # For BM25, we need to pre-fetch valid doc IDs to filter
        allowed_doc_ids = await self._get_filtered_doc_ids(section_filters, part_filters)

        bm25_results = self._bm25_search(query, candidate_pool_size, allowed_doc_ids)

        fused_doc_ids = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Fetch more documents if reranking
        fetch_count = candidate_pool_size if should_rerank else top_k
        documents = await self._fetch_documents_by_ids(fused_doc_ids[:fetch_count])

        if should_rerank:
            logger.info(f"Reranking {len(documents)} candidates")
            documents = reranker_service.rerank(query, documents, top_k)

        return documents[:top_k]

    async def get_by_section(self, section_reference: str) -> list[Document]:
        stmt = select(Document).where(
            Document.section_reference.ilike(f"%{section_reference}%")
        ).order_by(Document.page_number)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_document_count(self) -> int:
        result = await self.session.execute(text("SELECT COUNT(*) FROM documents"))
        return result.scalar()
