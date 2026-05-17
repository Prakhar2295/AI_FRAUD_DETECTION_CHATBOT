"""Fraud similarity scoring and semantic retrieval utilities."""
from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger


class FraudSimilaritySearch:
    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.logger = get_logger(self.__class__.__name__)

    def filter_results(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = [hit for hit in hits if hit.get("similarity", 0.0) >= self.threshold]
        self.logger.info("Filtered %s similarity hits above threshold=%s", len(results), self.threshold)
        return results

    def build_metadata(self, hits: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "retrieved_count": len(hits),
            "threshold": self.threshold,
            "top_matches": [hit.get("metadata", {}) for hit in hits[:3]],
        }
