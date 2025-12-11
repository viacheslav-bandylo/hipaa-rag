from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SourceReference(BaseModel):
    section_reference: str
    content: str
    page_number: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    session_id: str


class IngestRequest(BaseModel):
    pdf_path: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    chunks_processed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_model: str
