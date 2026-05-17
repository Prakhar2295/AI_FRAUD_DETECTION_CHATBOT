"""Detect urgency and forced-action language in conversation text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

URGENCY_PATTERNS = [
    "act immediately",
    "urgent transfer",
    "account will be blocked",
    "do not tell anyone",
    "right away",
    "before it's too late",
    "limited time",
    "immediately",
    "now",
    "asap",
]


@dataclass
class UrgencyResult:
    urgency_score: float
    details: Dict[str, Any]


class UrgencyDetector:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def analyze(self, transcript: str) -> UrgencyResult:
        text = transcript.lower()
        matches = [phrase for phrase in URGENCY_PATTERNS if phrase in text]
        score = min(1.0, len(matches) * 0.25 * self.weight)
        return UrgencyResult(urgency_score=score, details={"matches": matches})
