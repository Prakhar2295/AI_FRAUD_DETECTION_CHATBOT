"""Conversation memory workflow node."""

from __future__ import annotations

from typing import Any

from app.graph.state import FraudWorkflowState, append_trace, utc_now_iso
from app.models.conversation_models import ConversationInteraction
from app.services.memory_service import MemoryService
from app.utils.logger import get_logger

logger = get_logger("MemoryNode")


def create_memory_node(memory_service: MemoryService):
    """Create a node that persists the interaction into session memory."""

    def memory_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing memory node")
        updates = append_trace(state, "memory_node")
        trace = updates["workflow_trace"]

        interaction = ConversationInteraction(
            timestamp=utc_now_iso(),
            transcript=state["transcript"],
            intent=state.get("intent"),
            fraud=state.get("fraud"),
            risk=state.get("risk"),
            behavioral=state.get("behavioral"),
            fraud_audio=state.get("fraud_audio"),
            behavioral_metadata=state.get("behavioral_metadata"),
            retrieved_fraud_patterns=state.get("retrieved_fraud_patterns", []),
            adaptive_risk_metadata=state.get("adaptive_risk_enrichment"),
            fraud_knowledge_context=state.get("fraud_knowledge_context"),
            workflow_trace=trace,
        )
        session = memory_service.append_interaction(state["session_id"], interaction)

        return {
            **updates,
            "conversation_history": [_model_to_dict(item) for item in session.interactions],
            "session_metadata": {
                "session_id": session.session_id,
                "interaction_count": len(session.interactions),
            },
            "completed_at": utc_now_iso(),
        }

    return memory_node


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
