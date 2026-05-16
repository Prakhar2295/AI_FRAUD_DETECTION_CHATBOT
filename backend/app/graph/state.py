"""Typed LangGraph state for Phase 2 fraud intelligence workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict


class FraudWorkflowState(TypedDict, total=False):
    """Serializable state passed between LangGraph nodes."""

    session_id: str
    transcript: str
    partial_transcript: str | None
    accumulated_transcript: str | None
    stream_sequence: int | None
    intent: dict[str, Any] | None
    fraud: dict[str, Any] | None
    risk: dict[str, Any] | None
    suspicious_indicators: list[str]
    conversation_history: list[dict[str, Any]]
    workflow_history: list[dict[str, Any]]
    workflow_trace: list[str]
    node_timestamps: dict[str, str]
    started_at: str
    completed_at: str | None
    errors: list[str]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def create_initial_state(
    session_id: str,
    transcript: str,
    partial_transcript: str | None = None,
    stream_sequence: int | None = None,
    workflow_history: list[dict[str, Any]] | None = None,
) -> FraudWorkflowState:
    """Create an initial workflow state for one transcript."""
    return {
        "session_id": session_id,
        "transcript": transcript,
        "partial_transcript": partial_transcript,
        "accumulated_transcript": transcript,
        "stream_sequence": stream_sequence,
        "intent": None,
        "fraud": None,
        "risk": None,
        "suspicious_indicators": [],
        "conversation_history": [],
        "workflow_history": workflow_history or [],
        "workflow_trace": [],
        "node_timestamps": {},
        "started_at": utc_now_iso(),
        "completed_at": None,
        "errors": [],
    }


def append_trace(state: FraudWorkflowState, node_name: str) -> dict[str, Any]:
    """Return explicit trace and timestamp updates for a node."""
    trace = [*state.get("workflow_trace", []), node_name]
    timestamps = {**state.get("node_timestamps", {}), node_name: utc_now_iso()}
    return {"workflow_trace": trace, "node_timestamps": timestamps}
