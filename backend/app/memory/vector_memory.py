"""Vector fraud memory abstraction."""

from __future__ import annotations

from typing import Any

from app.vectorstore.chroma_manager import ChromaManager
from app.vectorstore.embedding_service import EmbeddingService
from app.utils.logger import get_logger


class VectorMemory:
    """Persist and retrieve fraud memory documents through a swappable vector store."""

    def __init__(
        self,
        vector_store: ChromaManager | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.vector_store = vector_store or ChromaManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = get_logger(self.__class__.__name__)

    def upsert(self, document_id: str, text: str, metadata: dict[str, Any]) -> None:
        embedding = self.embedding_service.embed_text(text)
        self.vector_store.upsert_document(
            document_id=document_id,
            content=text,
            metadata=metadata,
            embedding=embedding,
        )
        self.logger.info("Vector memory upserted document_id=%s", document_id)

    def search(self, text: str, top_k: int = 4) -> list[dict[str, Any]]:
        embedding = self.embedding_service.embed_text(text)
        return self.vector_store.query(
            query_text=text,
            n_results=top_k,
            query_embedding=embedding,
        )

