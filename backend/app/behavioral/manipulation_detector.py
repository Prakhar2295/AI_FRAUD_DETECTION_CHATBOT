"""Detect conversational manipulation and social engineering indicators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

MANIPULATION_PATTERNS = [
    "trusted source",
    "official",
    "authority",
    "verify your information",
    "confirm your identity",
    "this is confidential",
    "don't share",
    "only for you",
    "you are selected",
    "please cooperate",
]


@dataclass
class ManipulationResult:
    manipulation_confidence: float
    details: Dict[str, Any]


class ManipulationDetector:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def analyze(self, transcript: str) -> ManipulationResult:
        text = transcript.lower()
        matches = [phrase for phrase in MANIPULATION_PATTERNS if phrase in text]
        score = min(1.0, len(matches) * 0.2 * self.weight)
        return ManipulationResult(
            manipulation_confidence=score,
            details={"matches": matches},
        )
