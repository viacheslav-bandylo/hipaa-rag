import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import ChatRequest, ChatResponse, SourceReference
from app.services.retrieval import RetrievalService
from app.services.llm import llm_service


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session)
):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    retrieval_service = RetrievalService(session)

    doc_count = await retrieval_service.get_document_count()
    if doc_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please run the ingestion process first via POST /api/ingest"
        )

    if request.use_hybrid_search:
        relevant_docs = await retrieval_service.hybrid_search(request.message)
    else:
        relevant_docs = await retrieval_service.search(request.message)

    if not relevant_docs:
        return ChatResponse(
            answer="I couldn't find relevant information in the HIPAA documentation for your query. Please try rephrasing your question.",
            sources=[],
            session_id=session_id
        )

    answer = llm_service.generate_response(request.message, relevant_docs)

    sources = [
        SourceReference(
            section_reference=doc.section_reference,
            content=doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
            page_number=doc.page_number
        )
        for doc in relevant_docs
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id
    )


@router.get("/sections/{section_ref}")
async def get_section(
    section_ref: str,
    session: AsyncSession = Depends(get_session)
):
    retrieval_service = RetrievalService(session)
    documents = await retrieval_service.get_by_section(section_ref)

    if not documents:
        raise HTTPException(
            status_code=404,
            detail=f"No content found for section reference: {section_ref}"
        )

    return {
        "section_reference": section_ref,
        "chunks": [
            {
                "content": doc.content,
                "page_number": doc.page_number,
                "parent_context": doc.parent_context
            }
            for doc in documents
        ]
    }
