import logging
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.database import get_session, Document
from app.models import IngestRequest, IngestResponse
from app.services.pdf_parser import HIPAAParser
from app.services.embeddings import embedding_service
from app.services.bm25_service import bm25_service
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest_document(
        request: IngestRequest = None,
        session: AsyncSession = Depends(get_session)
):
    pdf_path = request.pdf_path if request and request.pdf_path else settings.pdf_path

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found at {pdf_path}. Please ensure the HIPAA PDF is placed in the data directory."
        )

    # 1. Clear existing data
    existing_count = await session.execute(text("SELECT COUNT(*) FROM documents"))
    if existing_count.scalar() > 0:
        await session.execute(text("TRUNCATE TABLE documents RESTART IDENTITY"))
        await session.commit()

    # 2. Parse PDF in ThreadPool (non-blocking) - FIX for CI-02
    parser = HIPAAParser(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        respect_list_boundaries=settings.respect_list_boundaries,
        max_list_chunk_size=settings.max_list_chunk_size
    )

    # Run CPU-bound parsing in threadpool to avoid blocking event loop
    chunks = await run_in_threadpool(parser.parse_pdf, pdf_path)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No content could be extracted from the PDF"
        )

    # 3. Generate Embeddings - DECOUPLED architecture (FIX for CI-03)
    # Concatenate context header with content for embeddings,
    # but store pure content in DB for clean data
    texts_for_embedding = []
    for chunk in chunks:
        # Combine context + content for richer semantic embedding
        if chunk.context_header:
            embedding_text = f"{chunk.context_header}\n{chunk.content}"
        else:
            embedding_text = chunk.content
        texts_for_embedding.append(embedding_text)

    # Run embedding generation in threadpool (CPU/GPU bound)
    embeddings = await run_in_threadpool(embedding_service.embed_texts, texts_for_embedding)

    # 4. Insert into Database - store PURE content (no metadata prefix)
    for chunk, embedding in zip(chunks, embeddings):
        # Handle page numbers: Parser returns List[int], DB expects int
        primary_page = chunk.page_numbers[0] if chunk.page_numbers else 0

        # Construct a useful reference string from available metadata
        ref_str = f"{chunk.part} | {chunk.subpart} | {chunk.section}"

        doc = Document(
            content=chunk.content,  # Pure text only, no metadata prefix
            part_number=chunk.part,
            section_number=chunk.section,
            section_reference=ref_str,
            paragraph_reference=chunk.paragraph_reference,
            page_number=primary_page,
            # Store context header separately for clean architecture
            parent_context=chunk.context_header or chunk.section_title,
            embedding=embedding
        )
        session.add(doc)

    await session.commit()

    # 5. Rebuild BM25 index with new documents
    # Include parent_context for better keyword matching on section references
    result = await session.execute(
        text("SELECT id, content, parent_context FROM documents ORDER BY id")
    )
    rows = result.fetchall()
    # Combine parent_context + content for BM25 indexing (same as embeddings)
    documents_for_bm25 = [
        (row.id, f"{row.parent_context}\n{row.content}" if row.parent_context else row.content)
        for row in rows
    ]
    bm25_count = bm25_service.build_index(documents_for_bm25)
    logger.info(f"BM25 index rebuilt with {bm25_count} documents after ingestion")

    return IngestResponse(
        status="success",
        chunks_processed=len(chunks),
        message=f"Successfully ingested {len(chunks)} chunks from HIPAA documentation"
    )


@router.get("/status")
async def get_ingestion_status(session: AsyncSession = Depends(get_session)):
    result = await session.execute(text("SELECT COUNT(*) FROM documents"))
    count = result.scalar()

    return {
        "indexed_chunks": count,
        "status": "ready" if count > 0 else "not_indexed",
        "pdf_path": settings.pdf_path,
        "pdf_exists": os.path.exists(settings.pdf_path)
    }
