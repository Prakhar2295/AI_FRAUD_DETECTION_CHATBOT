"""Semantic fraud retrieval workflow node."""
from __future__ import annotations

from typing import Any

from app.graph.state import FraudWorkflowState, append_trace
from app.utils.logger import get_logger
from app.services.memory_service import MemoryService

logger = get_logger("RetrievalNode")


def create_retrieval_node(memory_service: MemoryService):
    """Create a workflow node that retrieves similar fraud scenarios."""

    def retrieval_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing retrieval node")
        updates = append_trace(state, "retrieval_node")

        try:
            context = memory_service.retrieve_similar_fraud(
                session_id=state["session_id"],
                transcript=state["transcript"],
            )
            return {
                **updates,
                "retrieved_fraud_patterns": context.get("retrieved_fraud_patterns", []),
                "semantic_retrieval_metadata": context.get("semantic_retrieval_metadata", {}),
                "historical_fraud_context": context.get("historical_fraud_context", {}),
                "fraud_knowledge_context": context.get("fraud_knowledge_context", {}),
            }
        except Exception as exc:
            logger.error("Retrieval node failed: %s", exc)
            return {
                **updates,
                "errors": [*state.get("errors", []), f"retrieval_node: {exc}"],
                "retrieved_fraud_patterns": [],
                "semantic_retrieval_metadata": {},
                "historical_fraud_context": {},
                "fraud_knowledge_context": {},
            }

    return retrieval_node
