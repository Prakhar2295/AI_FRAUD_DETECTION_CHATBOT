"""Risk enrichment engine for adaptive fraud score adjustments."""
from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger


class RiskEnrichmentEngine:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def enrich(
        self,
        fraud_risk_score: int,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        boost = 0
        if context.get("pattern_matches"):
            boost += 5 * len(context["pattern_matches"])
        if context.get("retrieved_fraud_patterns"):
            boost += 3 * len(context["retrieved_fraud_patterns"])
        enriched_score = min(100, fraud_risk_score + boost)
        self.logger.info("Risk enrichment applied base=%s boost=%s result=%s", fraud_risk_score, boost, enriched_score)
        return {
            "enriched_score": enriched_score,
            "enrichment_reason": "knowledge_based_retrieval",
            "context": context,
        }
