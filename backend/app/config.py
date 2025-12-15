from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://hipaa:hipaa_secure_pass@localhost:5432/hipaa_rag"
    anthropic_api_key: str = ""

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_size: int = 800
    chunk_overlap: int = 100

    # Semantic chunking settings
    respect_list_boundaries: bool = True
    max_list_chunk_size: int = 2000  # Override limit for lists

    retrieval_top_k: int = 8

    # Reranker settings
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_candidates: int = 20  # Pre-rerank pool size

    llm_model: str = "claude-sonnet-4.5"
    llm_max_tokens: int = 10000

    pdf_path: str = "/app/data/HIPAA_questions.pdf"

    class Config:
        env_file = ".env"


settings = Settings()
