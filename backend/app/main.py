import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.routers import chat, ingest
from app.models import HealthResponse
from app.config import settings
from app.database import engine, async_session
from app.services.bm25_service import bm25_service

logger = logging.getLogger(__name__)


async def initialize_bm25_index():
    """Load all documents from DB and build BM25 index."""
    async with async_session() as session:
        result = await session.execute(
            text("SELECT id, content FROM documents ORDER BY id")
        )
        rows = result.fetchall()

        if rows:
            documents = [(row.id, row.content) for row in rows]
            count = bm25_service.build_index(documents)
            logger.info(f"BM25 index built with {count} documents")
        else:
            logger.info("No documents found, BM25 index not built")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_bm25_index()
    yield
    await engine.dispose()


app = FastAPI(
    title="HIPAA RAG API",
    description="Retrieval-Augmented Generation API for HIPAA documentation queries",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(ingest.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        database="postgresql",
        embedding_model=settings.embedding_model
    )


@app.get("/")
async def root():
    return {
        "service": "HIPAA RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
