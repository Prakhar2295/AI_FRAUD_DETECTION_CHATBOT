"""Deepfake voice analysis abstraction and lightweight checks.

This module provides a `DeepfakeDetector` with a placeholder implementation
that inspects spectral irregularities. Replace with model-based detectors in future.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class DeepfakeResult:
    deepfake_confidence: float
    details: Dict[str, Any]


class DeepfakeDetector:
    def __init__(self, threshold: float = 0.6):
        self.threshold = float(threshold)

    def analyze(self, pcm: np.ndarray, sr: int) -> DeepfakeResult:
        if pcm.size < 1024:
            return DeepfakeResult(deepfake_confidence=0.0, details={"reason": "too_short"})

        # Compute simple high-frequency energy ratio as a weak indicator
        spec = np.abs(np.fft.rfft(pcm.astype(float)))
        freqs = np.fft.rfftfreq(len(pcm), 1.0 / sr)
        hf_mask = freqs > (sr * 0.4)
        hf_energy = spec[hf_mask].sum() if hf_mask.any() else 0.0
        total_energy = spec.sum() + 1e-8
        hf_ratio = hf_energy / total_energy
        # synthetic voices sometimes show abnormal HF structure
        confidence = float(min(1.0, hf_ratio * 10.0))
        return DeepfakeResult(deepfake_confidence=confidence, details={"hf_ratio": float(hf_ratio)})
