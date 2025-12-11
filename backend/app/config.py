from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://hipaa:hipaa_secure_pass@localhost:5432/hipaa_rag"
    anthropic_api_key: str = ""

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_size: int = 800
    chunk_overlap: int = 100

    retrieval_top_k: int = 8

    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 2048

    pdf_path: str = "/app/data/HIPAA_questions.pdf"

    class Config:
        env_file = ".env"


settings = Settings()
