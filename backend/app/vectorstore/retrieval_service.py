"""High-level retrieval service for fraud memory search."""
from __future__ import annotations

from typing import Any

from app.vectorstore.chroma_manager import ChromaManager
from app.vectorstore.embedding_service import EmbeddingService
from app.utils.logger import get_logger


class RetrievalService:
    def __init__(
        self,
        vector_store: ChromaManager | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.vector_store = vector_store or ChromaManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = get_logger(self.__class__.__name__)

    def query(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        self.logger.info("Querying retrieval service for text length=%s", len(text))
        embedding = self.embedding_service.embed_text(text)
        return self.vector_store.query(
            query_text=text,
            n_results=n_results,
            query_embedding=embedding,
        )

    def embed(self, text: str) -> list[float]:
        return self.embedding_service.embed_text(text)
