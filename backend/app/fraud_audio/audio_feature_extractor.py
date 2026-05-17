"""Audio feature extraction helpers for Phase 5.

Provides lightweight, import-safe wrappers around `librosa` when available,
and fallback implementations using `numpy` for basic operations so tests can run
without heavy deps during early development.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional

try:
    import librosa
except Exception:  # pragma: no cover - librosa optional in dev
    librosa = None


@dataclass
class AudioFeatures:
    mfcc: Optional[np.ndarray]
    spectral_centroid: Optional[np.ndarray]
    zero_crossing_rate: Optional[np.ndarray]


def extract_mfcc(pcm: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    if librosa is not None:
        return librosa.feature.mfcc(y=pcm.astype(float), sr=sr, n_mfcc=n_mfcc)
    # fallback: simple framing + DCT placeholder (not production-grade)
    framed = _frame_signal(pcm, frame_size=1024, hop_length=512)
    # compute simple energy as placeholder for MFCC-like features
    energies = np.log(np.maximum(1e-8, (framed ** 2).sum(axis=1)))
    # tile to match n_mfcc
    return np.tile(energies[:, None], (1, n_mfcc)).T


def extract_spectral_centroid(pcm: np.ndarray, sr: int) -> np.ndarray:
    if librosa is not None:
        return librosa.feature.spectral_centroid(y=pcm.astype(float), sr=sr)
    framed = _frame_signal(pcm, frame_size=1024, hop_length=512)
    freqs = np.fft.rfftfreq(1024, 1.0 / sr)
    mags = np.abs(np.fft.rfft(framed, axis=1))
    centroid = (freqs * mags).sum(axis=1) / (mags.sum(axis=1) + 1e-8)
    return centroid[None, :]


def extract_zero_crossing_rate(pcm: np.ndarray) -> np.ndarray:
    if librosa is not None:
        return librosa.feature.zero_crossing_rate(y=pcm.astype(float))
    zcr = ((pcm[:-1] * pcm[1:]) < 0).astype(float)
    # return single-rate ZCR
    return np.array([zcr.mean()])


def _frame_signal(pcm: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    n_frames = 1 + max(0, (len(pcm) - frame_size) // hop_length)
    frames = np.stack([pcm[i * hop_length : i * hop_length + frame_size] for i in range(n_frames)])
    return frames


def extract_features(pcm: np.ndarray, sr: int) -> AudioFeatures:
    mfcc = extract_mfcc(pcm, sr)
    centroid = extract_spectral_centroid(pcm, sr)
    zcr = extract_zero_crossing_rate(pcm)
    return AudioFeatures(mfcc=mfcc, spectral_centroid=centroid, zero_crossing_rate=zcr)
