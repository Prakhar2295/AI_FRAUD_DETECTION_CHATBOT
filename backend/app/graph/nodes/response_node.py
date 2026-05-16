"""AI response generation workflow node."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.state import FraudWorkflowState, append_trace
from app.models.response_models import AIResponse
from app.services.llm_service import OllamaLLMService
from app.utils.logger import get_logger

logger = get_logger("ResponseNode")


def create_response_node(llm_service: OllamaLLMService):
    """Create a LangGraph node that generates a concise spoken agent response."""

    def response_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing response node")
        prompt = _build_prompt(state)

        try:
            payload = llm_service.generate_response(prompt)
            ai_response = AIResponse(**payload)
            updates = append_trace(state, "response_node")
            return {
                **updates,
                "ai_response": _model_to_dict(ai_response),
                "response_metadata": {
                    "voice_style": ai_response.voice_style,
                    "source": "fraud_agent",
                },
            }
        except (RuntimeError, TypeError, ValidationError) as exc:
            logger.error("Response node failed: %s", exc)
            fallback = AIResponse(
                response_text="I could not generate a spoken response right now. Please hold while I continue monitoring the conversation.",
                voice_style="neutral",
                metadata={"error": "response_generation_failed"},
            )
            updates = append_trace(state, "response_node")
            return {
                **updates,
                "ai_response": _model_to_dict(fallback),
                "response_metadata": {"voice_style": fallback.voice_style, "source": "fallback"},
                "errors": [*state.get("errors", []), f"response_node: {exc}"],
            }

    return response_node


def _build_prompt(state: FraudWorkflowState) -> str:
    previous = state.get("conversation_history", []) or []
    fraud = state.get("fraud") or {}
    risk = state.get("risk") or {}
    intent = state.get("intent") or {}
    transcript = state["transcript"]

    return (
        "You are a banking fraud investigation assistant speaking directly to the customer.\n"
        "Use the transcript, intent, fraud findings, and risk score to generate a concise spoken response.\n"
        "Keep the response short, natural, and suitable for voice playback.\n"
        "Do not use long paragraphs.\n"
        "Return only valid JSON with this exact schema:\n"
        "{\n"
        "  \"response_text\": \"spoken response text\",\n"
        "  \"voice_style\": \"neutral | reassuring | urgent | escalation\",\n"
        "  \"metadata\": {\"source\": \"fraud_agent\", \"relevant_indicators\": []}\n"
        "}\n\n"
        "Transcript:\n\"\"\"{transcript}\"\"\"\n\n"
        "Intent:\n{intent}\n\n"
        "Fraud analysis:\n{fraud}\n\n"
        "Risk summary:\n{risk}\n\n"
        "Conversation history:\n{previous}"
    )


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
