"""Extract conversational behavior signals from transcripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ConversationalSignals:
    pacing: float
    interruption_count: int
    repetition_count: int
    escalation_score: float
    instability_score: float


def extract_conversational_signals(transcript: str) -> ConversationalSignals:
    text = transcript.lower()
    tokens = text.split()
    token_count = len(tokens)
    punctuation_count = sum(1 for char in text if char in ".?!")

    pacing = min(1.0, (token_count / 50.0))
    interruption_count = text.count("...") + text.count("-")
    repetition_count = sum(tokens.count(word) - 1 for word in set(tokens) if tokens.count(word) > 1)
    escalation_score = min(1.0, punctuation_count / 5.0)
    instability_score = min(1.0, repetition_count / max(1, token_count) * 2.0)

    return ConversationalSignals(
        pacing=pacing,
        interruption_count=int(interruption_count),
        repetition_count=int(repetition_count),
        escalation_score=escalation_score,
        instability_score=instability_score,
    )
