import re
import string
from typing import Optional

from rank_bm25 import BM25Okapi


class BM25Service:
    """Singleton BM25 index service for keyword-based search."""

    _instance: Optional["BM25Service"] = None

    def __new__(cls) -> "BM25Service":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._bm25: Optional[BM25Okapi] = None
        self._document_ids: list[int] = []
        self._tokenized_corpus: list[list[str]] = []

    @property
    def is_indexed(self) -> bool:
        """Check if the BM25 index has been built."""
        return self._bm25 is not None and len(self._document_ids) > 0

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing/searching.

        Applies lowercase, removes punctuation, splits on whitespace.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def build_index(self, documents: list[tuple[int, str]]) -> int:
        """Build BM25 index from documents.

        Args:
            documents: List of (document_id, content) tuples

        Returns:
            Number of documents indexed
        """
        if not documents:
            self._bm25 = None
            self._document_ids = []
            self._tokenized_corpus = []
            return 0

        self._document_ids = [doc_id for doc_id, _ in documents]
        self._tokenized_corpus = [self.tokenize(content) for _, content in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        return len(self._document_ids)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search the BM25 index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document_id, bm25_score) tuples, sorted by score descending
        """
        if not self.is_indexed:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        doc_scores = list(zip(self._document_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def clear(self):
        """Clear the BM25 index."""
        self._bm25 = None
        self._document_ids = []
        self._tokenized_corpus = []


bm25_service = BM25Service()
