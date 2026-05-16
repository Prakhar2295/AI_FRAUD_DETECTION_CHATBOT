"""Pydantic response models for the voice intelligence pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]


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
    risk_level: RiskLevel
    suspicious_indicators: list[str]
    llm_reasoning: str


class IntentAnalysisResponse(BaseModel):
    """Structured intent classification result."""

    customer_intent: str
    transaction_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class FraudSignalAnalysisResponse(BaseModel):
    """Structured fraud signal analysis result."""

    suspicious_indicators: list[str] = Field(default_factory=list)
    urgency_manipulation: bool
    emotional_pressure: bool
    suspicious_intent: bool
    llm_reasoning: str


class RiskAnalysisResponse(BaseModel):
    """Deterministic risk score generated from workflow outputs."""

    fraud_risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    reasoning_summary: str


class WorkflowOutputResponse(BaseModel):
    """Final Phase 2 fraud intelligence output."""

    session_id: str
    transcript: str
    intent_classification: IntentAnalysisResponse | None
    suspicious_indicators: list[str]
    fraud_risk_score: int
    risk_level: RiskLevel
    reasoning_summary: str
    workflow_execution_trace: list[str]
    node_execution_timestamps: dict[str, str]
    conversation_turn_count: int
    errors: list[str] = Field(default_factory=list)
