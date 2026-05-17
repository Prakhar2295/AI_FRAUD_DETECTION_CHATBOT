"""Request and response models for the backend API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationAnalysisRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    transcript: str = Field(..., description="Transcript text to analyze")
    source: Optional[str] = Field(None, description="Optional source label")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., description="Query text to search against memory")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(0.5, description="Minimum similarity to include")
    session_id: Optional[str] = Field(None, description="Optional session context for retrieval")


class RetrievalMatch(BaseModel):
    similarity_score: float
    transcript: str
    risk_level: Optional[str]
    fraud_risk_score: Optional[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    matches: List[RetrievalMatch]
    retrieval_metadata: Dict[str, Any]


class MemorySessionResponse(BaseModel):
    session_id: str
    interaction_count: int
    history: List[Dict[str, Any]]


class AnalyticsResponse(BaseModel):
    active_sessions: int
    average_stt_latency_ms: float
    average_workflow_latency_ms: float
    average_retrieval_latency_ms: float
    memory_usage_bytes: int
    chroma_stats: Dict[str, Any]
