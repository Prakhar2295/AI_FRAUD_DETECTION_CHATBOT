"""Adaptive risk enrichment workflow node."""
from __future__ import annotations

from typing import Any

from app.graph.state import FraudWorkflowState, append_trace
from app.services.memory_service import MemoryService
from app.utils.logger import get_logger

logger = get_logger("AdaptiveRiskNode")


def create_adaptive_risk_node(memory_service: MemoryService):
    """Create a workflow node that enriches risk scoring using memory retrieval."""

    def adaptive_risk_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing adaptive risk node")
        updates = append_trace(state, "adaptive_risk_node")

        try:
            base_risk = state.get("risk") or {}
            retrieval_context = {
                "retrieved_fraud_patterns": state.get("retrieved_fraud_patterns", []),
                "historical_fraud_context": state.get("historical_fraud_context", {}),
            }
            enrichment = memory_service.enrich_risk(
                session_id=state["session_id"],
                base_risk=base_risk,
                retrieval_context=retrieval_context,
            )
            return {
                **updates,
                "risk": enrichment["risk"],
                "adaptive_risk_enrichment": enrichment["adaptive_risk_enrichment"],
            }
        except Exception as exc:
            logger.error("Adaptive risk node failed: %s", exc)
            fallback_risk = state.get("risk") or {"fraud_risk_score": 0, "risk_level": "low", "reasoning_summary": "adaptive_risk_failed"}
            return {
                **updates,
                "risk": fallback_risk,
                "adaptive_risk_enrichment": {"error": str(exc)},
                "errors": [*state.get("errors", []), f"adaptive_risk_node: {exc}"],
            }

    return adaptive_risk_node
