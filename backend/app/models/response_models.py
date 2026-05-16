"""Pydantic response models for the Phase 1 voice intelligence pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Structured response returned by the speech-to-text service."""

    audio_path: str
    text: str
    language: str | None = None
    duration_seconds: float | None = None


class FraudAnalysisResponse(BaseModel):
    """Structured banking fraud risk analysis result."""

    transcription: str
    fraud_risk_score: int = Field(ge=0, le=100)
    risk_level: str
    suspicious_indicators: list[str]
    llm_reasoning: str

