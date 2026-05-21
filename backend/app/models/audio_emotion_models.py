"""Pydantic models for audio emotion (SER) outputs."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class EmotionProbabilityModel(BaseModel):
    emotion: str
    probability: float = Field(ge=0.0, le=1.0)


class AudioBehavioralSignals(BaseModel):
    high_stress: bool = False
    emotional_instability: bool = False
    aggressive_tone: bool = False


class AudioEmotionResponse(BaseModel):
    dominant_emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    emotion_probabilities: Dict[str, float]
    audio_behavioral_signals: AudioBehavioralSignals
