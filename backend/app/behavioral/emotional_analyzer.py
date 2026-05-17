"""Detect emotional pressure and coercion in conversation text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

EMOTIONAL_PRESSURE_KEYWORDS = [
    "don't hesitate",
    "you must",
    "you have to",
    "this is urgent",
    "this is important",
    "if you care",
    "you need to",
    "asap",
    "right now",
    "immediately",
]


@dataclass
class EmotionalResult:
    emotional_risk_score: float
    details: Dict[str, Any]


class EmotionalAnalyzer:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def analyze(self, transcript: str) -> EmotionalResult:
        text = transcript.lower()
        count = sum(1 for phrase in EMOTIONAL_PRESSURE_KEYWORDS if phrase in text)
        score = min(1.0, count * 0.2 * self.weight)
        return EmotionalResult(
            emotional_risk_score=score,
            details={"pressure_phrases": count},
        )
