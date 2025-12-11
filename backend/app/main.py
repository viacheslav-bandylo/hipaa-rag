from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, ingest
from app.models import HealthResponse
from app.config import settings
from app.database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
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
