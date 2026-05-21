"""Map audio emotions into behavioral risk contributions."""
from __future__ import annotations

from typing import Dict, Any


class AudioRiskMapper:
    """Deterministic mapping from emotion probabilities to behavioral signals."""

    # Basic mapping rules; can be extended and tuned
    EMOTION_TO_SIGNALS: Dict[str, Dict[str, float]] = {
        "angry": {"high_stress": 0.9, "emotional_instability": 0.6, "aggressive_tone": 0.9},
        "fear": {"high_stress": 0.8, "emotional_instability": 0.7, "aggressive_tone": 0.2},
        "sad": {"high_stress": 0.6, "emotional_instability": 0.8, "aggressive_tone": 0.1},
        "neutral": {"high_stress": 0.0, "emotional_instability": 0.0, "aggressive_tone": 0.0},
        "happy": {"high_stress": 0.1, "emotional_instability": 0.0, "aggressive_tone": 0.0},
    }

    def map_emotions_to_signals(self, probs: Dict[str, float]) -> Dict[str, Any]:
        # initialize aggregated scores
        agg = {"high_stress": 0.0, "emotional_instability": 0.0, "aggressive_tone": 0.0}
        for emo, p in probs.items():
            key = emo.lower()
            mapping = self.EMOTION_TO_SIGNALS.get(key)
            if mapping:
                for signal, weight in mapping.items():
                    agg[signal] += weight * float(p)
            else:
                # unknown emotion: treat as neutral
                continue

        # threshold into booleans
        signals = {
            "high_stress": agg["high_stress"] >= 0.5,
            "emotional_instability": agg["emotional_instability"] >= 0.45,
            "aggressive_tone": agg["aggressive_tone"] >= 0.5,
        }
        return signals
