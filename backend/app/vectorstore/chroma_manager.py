"""Chroma-backed vector store with deterministic in-memory fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:  # pragma: no cover
    chromadb = None
    ChromaSettings = None

from app.utils.logger import get_logger


class ChromaManager:
    def __init__(
        self,
        collection_name: str = "fraud_memory",
        persist_dir: Path | None = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.collection_name = collection_name
        self.persist_dir = persist_dir or Path(
            os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma")
        ).expanduser().resolve()
        self._fallback_docs: list[dict[str, Any]] = []
        self._fallback_embeddings: list[list[float]] = []
        self._init_chroma()

    def _init_chroma(self) -> None:
        if chromadb is None or ChromaSettings is None:
            self.logger.warning("ChromaDB is unavailable; falling back to in-memory vector store")
            self.client = None
            self.collection = None
            return

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info(
                "Initialized Chroma collection=%s path=%s",
                self.collection_name,
                self.persist_dir,
            )
        except Exception as exc:
            self.logger.error("Failed to initialize Chroma client: %s", exc)
            self.client = None
            self.collection = None

    def upsert_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        if self.collection is not None and embedding is not None:
            try:
                self.collection.upsert(
                    ids=[document_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata],
                )
                return
            except Exception as exc:
                self.logger.error("Chroma upsert failed: %s", exc)

        self._fallback_docs.append(
            {"id": document_id, "content": content, "metadata": metadata}
        )
        if embedding is not None:
            self._fallback_embeddings.append(embedding)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if self.collection is not None and query_embedding is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["distances", "metadatas", "documents"],
                )
                hits = []
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = float(results["distances"][0][i])
                    hits.append(
                        {
                            "id": doc_id,
                            "similarity": max(0.0, 1.0 - distance),
                            "metadata": results["metadatas"][0][i],
                            "document": results["documents"][0][i],
                        }
                    )
                return hits
            except Exception as exc:
                self.logger.error("Chroma query failed: %s", exc)
        return self._fallback_query(query_text, n_results, query_embedding)

    def _fallback_query(
        self,
        query_text: str,
        n_results: int,
        query_embedding: list[float] | None,
    ) -> list[dict[str, Any]]:
        if not self._fallback_docs or not self._fallback_embeddings:
            return []
        query_embedding = query_embedding or self._hash_to_vector(query_text)
        similarities = [self._cosine_similarity(query_embedding, vector) for vector in self._fallback_embeddings]
        best = sorted(
            zip(similarities, self._fallback_docs), key=lambda item: item[0], reverse=True
        )[:n_results]
        return [
            {"id": doc["id"], "similarity": float(score), "metadata": doc["metadata"]}
            for score, doc in best
        ]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    @staticmethod
    def _hash_to_vector(text: str, dim: int = 128) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest]
        return [values[i % len(values)] for i in range(dim)]
