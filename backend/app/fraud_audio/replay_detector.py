"""Lightweight replay detection heuristics and interface.

This module provides a pluggable `ReplayDetector` interface and a simple
heuristic implementation suitable for low-latency streaming checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class ReplayResult:
    replay_confidence: float
    details: Dict[str, Any]


class ReplayDetector:
    """Detect replay attacks using lightweight temporal and spectral checks."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = float(threshold)

    def analyze(self, pcm: np.ndarray, sr: int) -> ReplayResult:
        # Simple heuristic: compute autocorrelation of a short window and
        # measure repetition peaks. This is a placeholder for research-grade models.
        if pcm.size < 1024:
            return ReplayResult(replay_confidence=0.0, details={"reason": "too_short"})

        window = pcm[: min(len(pcm), sr * 2)]
        norm = window - window.mean()
        corr = np.correlate(norm, norm, mode="full")
        mid = len(corr) // 2
        ac = corr[mid + 1 : mid + sr // 2]
        peak = float(ac.max() / (corr[mid] + 1e-8))
        # interpret high periodic peaks as potential replay
        confidence = min(1.0, max(0.0, (peak - 0.3) * 1.5))
        return ReplayResult(replay_confidence=confidence, details={"peak": peak})
