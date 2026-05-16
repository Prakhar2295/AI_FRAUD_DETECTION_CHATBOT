"""Intent classification workflow node."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.state import FraudWorkflowState, append_trace
from app.models.response_models import IntentAnalysisResponse
from app.services.llm_service import OllamaLLMService
from app.utils.logger import get_logger

logger = get_logger("IntentNode")


def create_intent_node(llm_service: OllamaLLMService):
    """Create a LangGraph node that classifies user intent."""

    def intent_node(state: FraudWorkflowState) -> dict[str, Any]:
        transcript = state["transcript"]
        logger.info("Executing intent node")
        prompt = _build_prompt(transcript)

        try:
            payload = llm_service.generate_response(prompt)
            intent = IntentAnalysisResponse(**payload)
            updates = append_trace(state, "intent_node")
            return {**updates, "intent": _model_to_dict(intent)}
        except (RuntimeError, TypeError, ValidationError) as exc:
            logger.error("Intent node failed: %s", exc)
            fallback = IntentAnalysisResponse(
                customer_intent="unknown",
                transaction_type="unknown",
                confidence=0.0,
                reasoning="Intent classification failed; using safe fallback.",
            )
            updates = append_trace(state, "intent_node")
            return {
                **updates,
                "intent": _model_to_dict(fallback),
                "errors": [*state.get("errors", []), f"intent_node: {exc}"],
            }

    return intent_node


def _build_prompt(transcript: str) -> str:
    return f"""
Classify the customer intent in this banking voice transcript.

Return only valid JSON with this exact schema:
{{
  "customer_intent": "short intent label",
  "transaction_type": "transfer | balance_check | card_issue | account_access | complaint | support | unknown",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Transcript:
\"\"\"{transcript}\"\"\"
""".strip()


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

