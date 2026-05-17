"""Vector index lifecycle and collection management."""
from __future__ import annotations

from typing import Any

from app.vectorstore.chroma_manager import ChromaManager
from app.utils.logger import get_logger


class VectorIndexManager:
    def __init__(self, collection_name: str = "fraud_memory") -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.collection_name = collection_name
        self.manager = ChromaManager(collection_name=collection_name)

    def ensure_index(self) -> None:
        self.logger.info("Ensuring vector index exists for collection=%s", self.collection_name)
        # In this design, ChromaManager initializes the collection lazily.
        if self.manager.collection is None:
            self.logger.warning("Vector index is running in fallback mode")

    def reset_index(self) -> None:
        self.logger.info("Resetting vector index for collection=%s", self.collection_name)
        # This method can be extended for collection recreation and cleanup in production.
