"""Aggregate detector signals into a deterministic authenticity score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AggregationResult:
    authenticity_score: float
    components: Dict[str, float]
    metadata: Dict[str, Any]


def aggregate_signals(signals: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> AggregationResult:
    """Deterministically aggregate component confidences into a 0..1 authenticity score.

    `signals` expected keys: `replay`, `spoof`, `deepfake` each in 0..1 where higher means more likely malicious.
    We convert to authenticity = 1 - weighted_malicious_score.
    """
    default_weights = {"replay": 1.0, "spoof": 1.0, "deepfake": 1.0}
    w = default_weights if weights is None else {**default_weights, **weights}
    malicious = 0.0
    for k in ["replay", "spoof", "deepfake"]:
        malicious += w.get(k, 0.0) * float(signals.get(k, 0.0))
    max_weight = sum(w.values())
    normalized = malicious / (max_weight + 1e-8)
    authenticity = max(0.0, min(1.0, 1.0 - normalized))
    return AggregationResult(authenticity_score=authenticity, components={k: float(signals.get(k, 0.0)) for k in signals}, metadata={"normalized_malicious": normalized})
