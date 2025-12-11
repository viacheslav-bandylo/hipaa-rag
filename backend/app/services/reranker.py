import logging
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import settings


logger = logging.getLogger(__name__)


class RerankerService:
    """Cross-encoder reranker service for improving retrieval precision.

    Uses a cross-encoder model to score (query, document) pairs and rerank
    results from the initial retrieval stage.
    """

    _instance = None
    _model: CrossEncoder | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model_loaded(self) -> None:
        """Load the cross-encoder model if not already loaded."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {settings.reranker_model}")
            self._model = CrossEncoder(settings.reranker_model)
            logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int | None = None
    ) -> list[Any]:
        """Rerank documents based on cross-encoder scores.

        Args:
            query: The search query
            documents: List of Document objects with 'content' attribute
            top_k: Number of top results to return (default: return all, sorted)

        Returns:
            Reranked list of documents (top_k if specified)
        """
        if not documents:
            return []

        if not settings.reranker_enabled:
            logger.debug("Reranker disabled, returning documents unchanged")
            return documents[:top_k] if top_k else documents

        self._ensure_model_loaded()

        # Create (query, document_content) pairs for scoring
        pairs = [(query, doc.content) for doc in documents]

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Combine documents with scores and sort by score descending
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        reranked = [doc for doc, _ in doc_scores]

        if top_k:
            reranked = reranked[:top_k]

        logger.debug(f"Reranked {len(documents)} documents, returning top {len(reranked)}")

        return reranked

    @property
    def is_enabled(self) -> bool:
        """Check if reranker is enabled in settings."""
        return settings.reranker_enabled

    @property
    def model_name(self) -> str:
        """Get the configured model name."""
        return settings.reranker_model


# Singleton instance
reranker_service = RerankerService()
