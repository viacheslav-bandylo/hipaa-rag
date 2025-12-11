from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Document
from app.services.embeddings import embedding_service
from app.config import settings


class RetrievalService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def search(self, query: str, top_k: int = None) -> list[Document]:
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

    async def get_by_section(self, section_reference: str) -> list[Document]:
        stmt = select(Document).where(
            Document.section_reference.ilike(f"%{section_reference}%")
        ).order_by(Document.page_number)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_document_count(self) -> int:
        result = await self.session.execute(text("SELECT COUNT(*) FROM documents"))
        return result.scalar()
