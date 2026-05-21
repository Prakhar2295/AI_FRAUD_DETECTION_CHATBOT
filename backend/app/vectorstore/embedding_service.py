"""Embedding service for fraud and behavioral content."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional
    SentenceTransformer = None

from app.utils.logger import get_logger
from app.config.settings import load_settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None, dim: int | None = None, provider: str | None = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        settings = load_settings()
        self.provider = provider or settings.embedding_provider
        self.model_name = model_name or settings.embedding_model
        self.dimensions = dim or settings.embedding_dimensions
        self.model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        if self.provider.lower() != "sentence-transformers":
            self.logger.info("EmbeddingService: using deterministic fallback provider=%s", self.provider)
            self.model = None
            return

        if SentenceTransformer is None:
            self.logger.warning("EmbeddingService: sentence-transformers not installed; falling back to deterministic embeddings")
            self.model = None
            return

        try:
            self.logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded")
        except Exception as exc:
            self.logger.exception("Failed to load embedding model %s: %s", self.model_name, exc)
            self.model = None

    def embed_text(self, text: str) -> list[float]:
        # Prefer model if available
        if self.model is not None:
            try:
                emb = self.model.encode(text, normalize_embeddings=True)
                arr = np.asarray(emb, dtype=float)
                # ensure L2-normalized
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                return arr.tolist()
            except Exception as exc:
                self.logger.exception("EmbeddingService.encode failed: %s", exc)

        # deterministic fallback
        return self._deterministic_embedding(text)

    def _deterministic_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest]
        # tile/trim to dimensions
        return [values[i % len(values)] for i in range(self.dimensions)]
