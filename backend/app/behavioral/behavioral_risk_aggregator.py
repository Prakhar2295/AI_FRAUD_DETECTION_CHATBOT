"""Aggregate behavioral fraud signals into a single behavioral risk score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BehavioralAggregationResult:
    behavioral_risk_score: float
    urgency_score: float
    emotional_risk_score: float
    manipulation_confidence: float
    hesitation_score: float
    stress_score: float
    social_engineering_confidence: float
    metadata: Dict[str, Any]


def aggregate_behavioral_signals(signals: dict[str, float], weights: dict[str, float] | None = None) -> BehavioralAggregationResult:
    default_weights = {
        "emotional": 1.0,
        "urgency": 1.0,
        "manipulation": 1.0,
        "hesitation": 0.8,
        "stress": 0.8,
        "social_engineering": 1.0,
    }
    merged_weights = {**default_weights, **(weights or {})}
    weighted = sum(signals.get(key, 0.0) * merged_weights[key] for key in merged_weights)
    max_weight = sum(merged_weights.values())
    risk_score = max(0.0, min(1.0, weighted / (max_weight + 1e-8)))
    return BehavioralAggregationResult(
        behavioral_risk_score=risk_score,
        urgency_score=signals.get("urgency", 0.0),
        emotional_risk_score=signals.get("emotional", 0.0),
        manipulation_confidence=signals.get("manipulation", 0.0),
        hesitation_score=signals.get("hesitation", 0.0),
        stress_score=signals.get("stress", 0.0),
        social_engineering_confidence=signals.get("social_engineering", 0.0),
        metadata={"weights": merged_weights, "raw_signals": signals},
    )
