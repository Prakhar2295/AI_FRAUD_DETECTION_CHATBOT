"""Analyze hesitation and stress indicators in conversation text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

HESITATION_PATTERNS = ["um", "uh", "well", "you know", "like", "I think", "kind of"]
STRESS_PATTERNS = ["nervous", "worried", "anxious", "hesitate", "not sure", "concerned"]


@dataclass
class StressResult:
    hesitation_score: float
    stress_score: float
    details: Dict[str, Any]


class StressAnalyzer:
    def __init__(self, hesitation_weight: float = 1.0, stress_weight: float = 1.0):
        self.hesitation_weight = float(hesitation_weight)
        self.stress_weight = float(stress_weight)

    def analyze(self, transcript: str) -> StressResult:
        text = transcript.lower()
        hesitation_count = sum(text.count(token) for token in HESITATION_PATTERNS)
        stress_count = sum(text.count(token) for token in STRESS_PATTERNS)

        hesitation_score = min(1.0, hesitation_count * 0.15 * self.hesitation_weight)
        stress_score = min(1.0, stress_count * 0.2 * self.stress_weight)

        return StressResult(
            hesitation_score=hesitation_score,
            stress_score=stress_score,
            details={"hesitation_count": hesitation_count, "stress_count": stress_count},
        )
