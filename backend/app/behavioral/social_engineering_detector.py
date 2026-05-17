"""Detect social engineering fraud patterns in conversation text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

SOCIAL_ENGINEERING_PATTERNS = [
    "someone asked me to",
    "on behalf of",
    "trusted partner",
    "confidential information",
    "security reasons",
    "your account has been compromised",
    "I need your",
    "for verification purposes",
    "please keep this private",
    "secret code",
]


@dataclass
class SocialEngineeringResult:
    social_engineering_confidence: float
    details: Dict[str, Any]


class SocialEngineeringDetector:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def analyze(self, transcript: str) -> SocialEngineeringResult:
        text = transcript.lower()
        matches = [phrase for phrase in SOCIAL_ENGINEERING_PATTERNS if phrase in text]
        score = min(1.0, len(matches) * 0.2 * self.weight)
        return SocialEngineeringResult(
            social_engineering_confidence=score,
            details={"matches": matches},
        )
