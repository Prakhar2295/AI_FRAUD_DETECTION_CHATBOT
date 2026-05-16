"""Risk scoring workflow node."""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.graph.state import FraudWorkflowState, append_trace
from app.models.response_models import RiskAnalysisResponse
from app.utils.logger import get_logger

logger = get_logger("RiskNode")


def create_risk_node(settings: Settings):
    """Create a deterministic risk scoring node."""

    def risk_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing risk node")
        fraud = state.get("fraud") or {}
        suspicious_indicators = list(state.get("suspicious_indicators", []))

        score = _score_fraud_signals(fraud, suspicious_indicators)
        risk = RiskAnalysisResponse(
            fraud_risk_score=score,
            risk_level=_risk_level(score, settings),
            reasoning_summary=_reasoning_summary(fraud, suspicious_indicators, score),
        )

        updates = append_trace(state, "risk_node")
        return {**updates, "risk": _model_to_dict(risk)}

    return risk_node


def _score_fraud_signals(fraud: dict[str, Any], indicators: list[str]) -> int:
    score = min(len(indicators) * 15, 45)

    if fraud.get("urgency_manipulation"):
        score += 20
    if fraud.get("emotional_pressure"):
        score += 15
    if fraud.get("suspicious_intent"):
        score += 25

    return max(0, min(score, 100))


def _risk_level(score: int, settings: Settings) -> str:
    if score <= settings.low_risk_max_score:
        return "low"
    if score <= settings.medium_risk_max_score:
        return "medium"
    return "high"


def _reasoning_summary(
    fraud: dict[str, Any],
    indicators: list[str],
    score: int,
) -> str:
    if not indicators and score == 0:
        return "No strong fraud signals were detected in the transcript."

    base_reason = fraud.get("llm_reasoning") or "Fraud signals were detected."
    return f"{base_reason} Indicators found: {len(indicators)}."


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

