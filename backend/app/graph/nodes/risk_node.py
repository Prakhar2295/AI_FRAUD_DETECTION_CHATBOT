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
        behavioral = state.get("behavioral") or {}
        suspicious_indicators = list(state.get("suspicious_indicators", []))

        score = _score_fraud_signals(fraud, behavioral, suspicious_indicators)
        risk = RiskAnalysisResponse(
            fraud_risk_score=score,
            risk_level=_risk_level(score, settings),
            reasoning_summary=_reasoning_summary(fraud, behavioral, suspicious_indicators, score),
        )

        updates = append_trace(state, "risk_node")
        return {**updates, "risk": _model_to_dict(risk)}

    return risk_node


def _score_fraud_signals(fraud: dict[str, Any], behavioral: dict[str, Any], indicators: list[str]) -> int:
    score = min(len(indicators) * 15, 45)

    if fraud.get("urgency_manipulation"):
        score += 20
    if fraud.get("emotional_pressure"):
        score += 15
    if fraud.get("suspicious_intent"):
        score += 25

    if behavioral:
        score += int(behavioral.get("behavioral_risk_score", 0) * 40)
        score += int(behavioral.get("urgency_score", 0.0) * 10)
        score += int(behavioral.get("emotional_risk_score", 0.0) * 10)
        score += int(behavioral.get("manipulation_confidence", 0.0) * 10)
        score += int(behavioral.get("hesitation_score", 0.0) * 5)
        score += int(behavioral.get("stress_score", 0.0) * 5)
        score += int(behavioral.get("social_engineering_confidence", 0.0) * 10)

    return max(0, min(score, 100))


def _risk_level(score: int, settings: Settings) -> str:
    if score <= settings.low_risk_max_score:
        return "low"
    if score <= settings.medium_risk_max_score:
        return "medium"
    return "high"


def _reasoning_summary(
    fraud: dict[str, Any],
    behavioral: dict[str, Any],
    indicators: list[str],
    score: int,
) -> str:
    if not indicators and not behavioral and score == 0:
        return "No strong fraud signals were detected in the transcript."

    parts = []
    if fraud.get("llm_reasoning"):
        parts.append(fraud.get("llm_reasoning"))
    if behavioral:
        parts.append(
            "Behavioral analysis indicated risk signals:"
            f" urgency={behavioral.get('urgency_score', 0.0):.2f},"
            f" emotional={behavioral.get('emotional_risk_score', 0.0):.2f},"
            f" manipulation={behavioral.get('manipulation_confidence', 0.0):.2f}"
        )
    if indicators:
        parts.append(f"Indicators found: {len(indicators)}.")
    return " ".join(parts)


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

