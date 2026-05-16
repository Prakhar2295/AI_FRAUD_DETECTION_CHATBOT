"""WebSocket message models for realtime streaming."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.response_models import WorkflowOutputResponse


InboundMessageType = Literal["flush", "ping", "stop"]
OutboundMessageType = Literal[
    "session_started",
    "ack",
    "transcription_update",
    "fraud_intelligence",
    "session_state",
    "error",
    "session_closed",
]


class AudioChunkMetadata(BaseModel):
    """Metadata describing one inbound audio chunk."""

    sequence: int
    timestamp_ms: int | None = None
    sample_rate: int
    channels: int
    encoding: Literal["pcm_s16le", "wav"] = "pcm_s16le"


class InboundControlMessage(BaseModel):
    """JSON control message accepted by the realtime WebSocket endpoint."""

    type: InboundMessageType
    sequence: int | None = None
    timestamp_ms: int | None = None


class StreamingTranscriptionUpdate(BaseModel):
    """Partial transcript emitted after a chunk-window transcription."""

    session_id: str
    transcript: str
    accumulated_transcript: str
    sequence: int
    is_final: bool = False


class WebSocketOutboundMessage(BaseModel):
    """Typed outbound message sent to WebSocket clients."""

    type: OutboundMessageType
    session_id: str
    payload: dict = Field(default_factory=dict)


class StreamingFraudIntelligenceMessage(BaseModel):
    """Structured fraud intelligence emitted during streaming."""

    session_id: str
    sequence: int
    workflow_output: WorkflowOutputResponse
