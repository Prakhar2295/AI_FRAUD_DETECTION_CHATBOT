"""Fraud signal analysis workflow node."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.state import FraudWorkflowState, append_trace
from app.models.response_models import FraudSignalAnalysisResponse
from app.services.llm_service import OllamaLLMService
from app.utils.logger import get_logger

logger = get_logger("FraudNode")


def create_fraud_node(llm_service: OllamaLLMService):
    """Create a LangGraph node that detects suspicious fraud signals."""

    def fraud_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing fraud node")
        prompt = _build_prompt(state)

        try:
            payload = llm_service.generate_response(prompt)
            fraud = FraudSignalAnalysisResponse(**payload)
            updates = append_trace(state, "fraud_node")
            return {
                **updates,
                "fraud": _model_to_dict(fraud),
                "suspicious_indicators": fraud.suspicious_indicators,
            }
        except (RuntimeError, TypeError, ValidationError) as exc:
            logger.error("Fraud node failed: %s", exc)
            fallback = FraudSignalAnalysisResponse(
                suspicious_indicators=[],
                urgency_manipulation=False,
                emotional_pressure=False,
                suspicious_intent=False,
                llm_reasoning="Fraud signal analysis failed; using conservative fallback.",
            )
            updates = append_trace(state, "fraud_node")
            return {
                **updates,
                "fraud": _model_to_dict(fallback),
                "suspicious_indicators": [],
                "errors": [*state.get("errors", []), f"fraud_node: {exc}"],
            }

    return fraud_node


def _build_prompt(state: FraudWorkflowState) -> str:
    intent = state.get("intent") or {}
    return f"""
Analyze this banking voice transcript for fraud signals.

Focus on:
- urgency manipulation
- suspicious intent
- emotional pressure
- requests for OTP, PIN, CVV, passwords, account access, remote access, or immediate transfer
- impersonation of bank staff, law enforcement, government, courier, or support agent

Return only valid JSON with this exact schema:
{{
  "suspicious_indicators": ["indicator 1"],
  "urgency_manipulation": false,
  "emotional_pressure": false,
  "suspicious_intent": false,
  "llm_reasoning": "brief explanation"
}}

Intent context:
{intent}

Transcript:
\"\"\"{state["transcript"]}\"\"\"
""".strip()


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

