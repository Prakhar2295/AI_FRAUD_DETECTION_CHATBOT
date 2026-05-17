"""Behavioral fraud intelligence workflow node."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.state import FraudWorkflowState, append_trace
from app.models.response_models import BehavioralAnalysisResponse
from app.services.llm_service import OllamaLLMService
from app.utils.logger import get_logger

logger = get_logger("BehavioralNode")


def create_behavioral_node(llm_service: OllamaLLMService):
    """Create a LangGraph node that enriches behavioral fraud intelligence."""

    def behavioral_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing behavioral node")
        prompt = _build_prompt(state)

        try:
            payload = llm_service.generate_response(prompt)
            behavioral = BehavioralAnalysisResponse(**payload)
            updates = append_trace(state, "behavioral_node")
            return {
                **updates,
                "behavioral": _model_to_dict(behavioral),
            }
        except (RuntimeError, TypeError, ValidationError) as exc:
            logger.error("Behavioral node failed: %s", exc)
            fallback = BehavioralAnalysisResponse(
                behavioral_risk_score=0,
                urgency_score=0.0,
                emotional_risk_score=0.0,
                manipulation_indicators=[],
                hesitation_score=0.0,
                stress_score=0.0,
                social_engineering_confidence=0.0,
                metadata={"reason": "behavioral_node_failed"},
            )
            updates = append_trace(state, "behavioral_node")
            return {
                **updates,
                "behavioral": _model_to_dict(fallback),
                "errors": [*state.get("errors", []), f"behavioral_node: {exc}"],
            }

    return behavioral_node


def _build_prompt(state: FraudWorkflowState) -> str:
    transcript = state["transcript"]
    fraud_audio = state.get("fraud_audio") or {}
    behavioral_metadata = state.get("behavioral_metadata") or {}
    return f"""
Analyze this banking voice transcript for behavioral fraud signals and conversational manipulation.

Use the available contextual metadata from authenticity and behavioral analysis if present.

Return only valid JSON with this exact schema:
{{
  "behavioral_risk_score": 0,
  "urgency_score": 0.0,
  "emotional_risk_score": 0.0,
  "manipulation_indicators": [],
  "hesitation_score": 0.0,
  "stress_score": 0.0,
  "social_engineering_confidence": 0.0,
  "metadata": {{"summary": ""}}
}}

Transcript:
{transcript}

Fraud audio metadata:
{fraud_audio}

Behavioral metadata:
{behavioral_metadata}
""".strip()


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
