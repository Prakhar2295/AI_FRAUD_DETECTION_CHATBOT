"""Detect audio artifacts and corrupted frames.

Provides lightweight checks for clipping, DC offset, and NaN/Inf samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class ArtifactResult:
    clipped: bool
    dc_offset: float
    invalid_samples: int
    details: Dict[str, Any]


def analyze_artifacts(pcm: np.ndarray) -> ArtifactResult:
    if pcm.size == 0:
        return ArtifactResult(clipped=False, dc_offset=0.0, invalid_samples=0, details={})

    clipped = bool(np.any(np.abs(pcm) >= np.iinfo(pcm.dtype).max)) if np.issubdtype(pcm.dtype, np.integer) else bool(np.any(np.abs(pcm) > 0.9999))
    dc_offset = float(np.mean(pcm))
    invalid = int(np.isnan(pcm).sum() + np.isinf(pcm).sum())
    details = {"max": float(np.max(pcm)), "min": float(np.min(pcm))}
    return ArtifactResult(clipped=clipped, dc_offset=dc_offset, invalid_samples=invalid, details=details)
