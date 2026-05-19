"""Adaptive risk enrichment engine."""
from __future__ import annotations

from typing import Any

from app.memory.session_memory import SessionMemory
from app.utils.logger import get_logger


class AdaptiveRiskEngine:
    def __init__(
        self,
        session_memory: SessionMemory,
    ) -> None:
        self.session_memory = session_memory
        self.logger = get_logger(self.__class__.__name__)

    def enrich_risk(
        self,
        session_id: str,
        base_risk: dict[str, Any],
        retrieval_context: dict[str, Any],
    ) -> dict[str, Any]:
        self.logger.info("Enriching risk for session_id=%s", session_id)
        base_score = int(base_risk.get("fraud_risk_score", 0))
        history = self.session_memory.get_history(session_id)

        suspicious_history = [
            item
            for item in history
            if item.risk and item.risk.fraud_risk_score >= 60
        ]

        retrieval_boost = min(
            15,
            int(
                retrieval_context.get("retrieved_fraud_patterns", [])
                and len(retrieval_context["retrieved_fraud_patterns"]) * 4
                or 0
            )
        )

        history_penalty = min(15, len(suspicious_history) * 3)
        adjusted_score = min(100, base_score + retrieval_boost + history_penalty)

        if adjusted_score >= 75:
            risk_level = "high"
        elif adjusted_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        enrichment_metadata = {
            "base_score": base_score,
            "retrieval_boost": retrieval_boost,
            "history_penalty": history_penalty,
            "adjusted_score": adjusted_score,
            "adaptive_rules": [
                "retrieval_similarity_boost",
                "cross_session_activity_boost",
            ],
        }
        return {
            "risk": {
                "fraud_risk_score": adjusted_score,
                "risk_level": risk_level,
                "reasoning_summary": base_risk.get("reasoning_summary", "") + " | adaptive_enrichment_applied",
            },
            "adaptive_risk_enrichment": enrichment_metadata,
        }
