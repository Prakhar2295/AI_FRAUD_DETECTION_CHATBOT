"""Conversation and workflow models for Phase 2."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.models.response_models import (
    BehavioralAnalysisResponse,
    FraudSignalAnalysisResponse,
    IntentAnalysisResponse,
    RiskAnalysisResponse,
)


class ConversationInteraction(BaseModel):
    """One processed voice interaction in a conversation session."""

    timestamp: str
    transcript: str
    intent: IntentAnalysisResponse | None = None
    fraud: FraudSignalAnalysisResponse | None = None
    risk: RiskAnalysisResponse | None = None
    behavioral: BehavioralAnalysisResponse | None = None
    fraud_audio: dict[str, Any] | None = None
    behavioral_metadata: dict[str, Any] | None = None
    retrieved_fraud_patterns: list[dict[str, Any]] = Field(default_factory=list)
    adaptive_risk_metadata: dict[str, Any] | None = None
    fraud_knowledge_context: dict[str, Any] | None = None
    workflow_trace: list[str] = Field(default_factory=list)


class ConversationSession(BaseModel):
    """Session-scoped conversation memory container."""

    session_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    interactions: list[ConversationInteraction] = Field(default_factory=list)


class WorkflowStateModel(BaseModel):
    """Serializable snapshot of the LangGraph workflow state."""

    session_id: str
    transcript: str
    intent: IntentAnalysisResponse | None = None
    fraud: FraudSignalAnalysisResponse | None = None
    risk: RiskAnalysisResponse | None = None
    behavioral: BehavioralAnalysisResponse | None = None
    fraud_audio: dict[str, Any] | None = None
    behavioral_metadata: dict[str, Any] | None = None
    retrieved_fraud_patterns: list[dict[str, Any]] = Field(default_factory=list)
    semantic_retrieval_metadata: dict[str, Any] | None = None
    historical_fraud_context: dict[str, Any] | None = None
    adaptive_risk_enrichment: dict[str, Any] | None = None
    fraud_knowledge_context: dict[str, Any] | None = None
    suspicious_indicators: list[str] = Field(default_factory=list)
    conversation_history: list[ConversationInteraction] = Field(default_factory=list)
    workflow_trace: list[str] = Field(default_factory=list)
    node_timestamps: dict[str, str] = Field(default_factory=dict)
    started_at: str
    completed_at: str | None = None
    errors: list[str] = Field(default_factory=list)
