"""Persistent fraud retrieval engine."""
from __future__ import annotations

from typing import Any

from app.memory.fraud_pattern_store import FraudPatternStore
from app.vectorstore.chroma_manager import ChromaManager
from app.vectorstore.embedding_service import EmbeddingService
from app.vectorstore.fraud_similarity_search import FraudSimilaritySearch
from app.utils.logger import get_logger


class RetrievalEngine:
    def __init__(
        self,
        vector_store: ChromaManager,
        pattern_store: FraudPatternStore,
        embedding_service: EmbeddingService | None = None,
        top_k: int = 4,
        similarity_threshold: float = 0.55,
    ) -> None:
        self.vector_store = vector_store
        self.pattern_store = pattern_store
        self.embedding_service = embedding_service or EmbeddingService()
        self.top_k = top_k
        self.similarity_search = FraudSimilaritySearch(threshold=similarity_threshold)
        self.logger = get_logger(self.__class__.__name__)

    def retrieve_similar_fraud(
        self,
        transcript: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        self.logger.info("Retrieving similar fraud scenarios for transcript length=%s", len(transcript))
        settings = __import__("app.config.settings", fromlist=["load_settings"]).load_settings()
        threshold = getattr(settings, "vector_similarity_threshold", self.similarity_search.threshold)
        query_embedding = self.embedding_service.embed_text(transcript)
        vector_results = self.vector_store.query(
            query_text=transcript,
            n_results=top_k or self.top_k,
            query_embedding=query_embedding,
        )

        # apply similarity filter first
        vector_results = [r for r in vector_results if (r.get("similarity") or 0.0) >= threshold]
        vector_results = self.similarity_search.filter_results(vector_results)

        fraud_matches = []
        for hit in vector_results:
            metadata = hit.get("metadata") or {}
            fraud_score = metadata.get("fraud_risk_score")
            indicators = metadata.get("suspicious_indicators") or ""
            # filter low-quality matches: require fraud score and non-empty indicators
            try:
                fs = float(fraud_score) if fraud_score is not None else 0.0
            except Exception:
                fs = 0.0

            if fs <= 0.0:
                self.logger.debug("Dropping match with low fraud score: %s", hit.get("id"))
                continue
            if not indicators:
                self.logger.debug("Dropping match with empty suspicious indicators: %s", hit.get("id"))
                continue

            fraud_matches.append(
                {
                    "document_id": hit.get("id"),
                    "similarity": float(hit.get("similarity", 0.0)),
                    "metadata": metadata,
                    "document": hit.get("document"),
                    "reason": "similarity_and_fraud_score",
                }
            )

        pattern_matches = self.pattern_store.find_matching_patterns(transcript)
        enriched_metadata = {
            "vector_hit_count": len(fraud_matches),
            "pattern_hit_count": len(pattern_matches),
            "similarity_threshold": threshold,
        }

        return {
            "retrieved_fraud_patterns": fraud_matches,
            "semantic_retrieval_metadata": enriched_metadata,
            "historical_fraud_context": {},
            "fraud_knowledge_context": {
                "pattern_matches": pattern_matches,
            },
        }
