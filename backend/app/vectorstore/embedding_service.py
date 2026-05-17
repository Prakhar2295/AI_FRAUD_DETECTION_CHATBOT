"""Embedding service for fraud and behavioral content."""
from __future__ import annotations

import hashlib
import os
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from app.utils.logger import get_logger


class EmbeddingService:
    def __init__(
        self,
        model_name: str | None = None,
        dim: int = 128,
        provider: str | None = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.dimensions = dim
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "deterministic")
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if self.provider != "sentence-transformers":
            self.logger.info("Using deterministic local embedding provider")
            return None
        if SentenceTransformer is None:
            self.logger.warning("SentenceTransformer not installed; using deterministic embeddings")
            return None
        try:
            return SentenceTransformer(self.model_name)
        except Exception as exc:
            self.logger.error("Failed to load embedding model: %s", exc)
            return None

    def embed_text(self, text: str) -> list[float]:
        if self.model is not None:
            try:
                embedding = self.model.encode(text, normalize_embeddings=True)
                return [float(value) for value in embedding.tolist()]
            except Exception as exc:
                self.logger.error("Embedding model failed: %s", exc)
        return self._deterministic_embedding(text)

    def _deterministic_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest]
        return [values[i % len(values)] for i in range(self.dimensions)]
