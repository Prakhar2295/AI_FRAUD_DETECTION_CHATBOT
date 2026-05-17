"""Extract hesitation and uncertainty signals from transcripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

UNCERTAINTY_PATTERNS = ["not sure", "maybe", "perhaps", "I think", "I guess", "I believe"]


@dataclass
class HesitationResult:
    hesitation_score: float
    details: Dict[str, Any]


class HesitationAnalyzer:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def analyze(self, transcript: str) -> HesitationResult:
        text = transcript.lower()
        matches = sum(text.count(pattern) for pattern in UNCERTAINTY_PATTERNS)
        score = min(1.0, matches * 0.2 * self.weight)
        return HesitationResult(hesitation_score=score, details={"matches": matches})
