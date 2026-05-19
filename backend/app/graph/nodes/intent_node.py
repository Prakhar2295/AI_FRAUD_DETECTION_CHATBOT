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
You are a banking fraud intelligence analyst.

Analyze the banking voice transcript and classify:
1. customer intent
2. transaction category
3. possible fraud-oriented intent
4. confidence level

Focus especially on:
- OTP requests
- urgency manipulation
- credential collection
- impersonation attempts
- account access pressure
- suspicious social engineering patterns

Return ONLY valid JSON.

Schema:
{{
  "customer_intent": "otp_request | balance_inquiry | account_access | credential_collection | complaint | support | unknown",

  "transaction_type": "transfer | balance_check | card_issue | account_access | complaint | support | unknown",

  "fraud_intent_detected": true,

  "fraud_intent_type": "urgency_manipulation | impersonation | credential_theft | social_engineering | none",

  "confidence": 0.0,

  "reasoning": "brief explanation"
}}

Examples:

Transcript:
"Please share OTP immediately or your account will be blocked"

Output:
{{
  "customer_intent": "otp_request",
  "transaction_type": "account_access",
  "fraud_intent_detected": true,
  "fraud_intent_type": "urgency_manipulation",
  "confidence": 0.91,
  "reasoning": "The caller is urgently requesting OTP access using fear-based pressure."
}}

Transcript:
"What is my account balance?"

Output:
{{
  "customer_intent": "balance_inquiry",
  "transaction_type": "balance_check",
  "fraud_intent_detected": false,
  "fraud_intent_type": "none",
  "confidence": 0.97,
  "reasoning": "Legitimate banking inquiry."
}}

Now analyze this transcript:

\"\"\"{transcript}\"\"\"
""".strip()


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

