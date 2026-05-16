"""Application entry point for CLI and FastAPI realtime streaming."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.settings import Settings, load_settings
from app.api.websocket import create_websocket_router
from app.graph.fraud_workflow import build_fraud_workflow
from app.graph.state import create_initial_state
from app.models.response_models import RiskAnalysisResponse, WorkflowOutputResponse
from app.services.fraud_analysis_service import FraudAnalysisService
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.services.microphone_service import MicrophoneService
from app.services.stt_service import STTService
from app.utils.logger import get_logger


def _default_audio_path() -> Path:
    """Return the default Phase 1 sample audio path."""
    return BACKEND_ROOT / "data" / "fraud_detection_sample.wav"


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Support Pydantic v1 and v2 serialization."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def build_stt_service(settings: Settings) -> STTService:
    """Build the configured speech-to-text service."""
    return STTService(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def build_llm_service(settings: Settings) -> OllamaLLMService:
    """Build the configured Ollama service."""
    return OllamaLLMService(
        model_name=settings.ollama_model,
        endpoint=settings.ollama_endpoint,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def run_phase1_file_pipeline(audio_path: Path) -> dict[str, Any]:
    """Run the Phase 1 file-based pipeline for smoke testing and fallback usage."""
    settings = load_settings()
    stt_service = STTService(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    llm_service = OllamaLLMService(
        model_name=settings.ollama_model,
        endpoint=settings.ollama_endpoint,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    fraud_service = FraudAnalysisService(llm_service=llm_service)

    transcription = stt_service.transcribe_audio(audio_path)
    analysis = fraud_service.analyze_transcription(transcription.text)

    return {
        "transcription": _model_to_dict(transcription),
        "fraud_analysis": _model_to_dict(analysis),
    }


def run_live_voice_pipeline(session_id: str | None = None) -> WorkflowOutputResponse:
    """Capture microphone audio, transcribe it, and execute the LangGraph workflow."""
    settings = load_settings()
    logger = get_logger("main")
    session = session_id or _new_session_id()

    microphone_service = MicrophoneService(settings=settings)
    stt_service = build_stt_service(settings)
    llm_service = build_llm_service(settings)
    memory_service = MemoryService()

    audio_path = microphone_service.capture_audio()
    transcription = stt_service.transcribe_audio(audio_path)

    logger.info("Initializing LangGraph workflow: session_id=%s", session)
    workflow = build_fraud_workflow(
        llm_service=llm_service,
        memory_service=memory_service,
        settings=settings,
    )
    initial_state = create_initial_state(
        session_id=session,
        transcript=transcription.text,
    )
    final_state = workflow.invoke(initial_state)

    return _workflow_output(final_state)


def _workflow_output(state: dict[str, Any]) -> WorkflowOutputResponse:
    risk = RiskAnalysisResponse(**state["risk"])
    history = state.get("conversation_history", [])
    return WorkflowOutputResponse(
        session_id=state["session_id"],
        transcript=state["transcript"],
        partial_transcript=state.get("partial_transcript"),
        stream_sequence=state.get("stream_sequence"),
        intent_classification=state.get("intent"),
        suspicious_indicators=state.get("suspicious_indicators", []),
        fraud_risk_score=risk.fraud_risk_score,
        risk_level=risk.risk_level,
        reasoning_summary=risk.reasoning_summary,
        workflow_execution_trace=state.get("workflow_trace", []),
        node_execution_timestamps=state.get("node_timestamps", {}),
        conversation_turn_count=len(history),
        errors=state.get("errors", []),
    )


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"local-session-{timestamp}"


def create_app() -> FastAPI:
    """Create the FastAPI application with realtime WebSocket routes."""
    settings = load_settings()
    app = FastAPI(title="Banking Fraud Detection Voice AI")
    app.include_router(create_websocket_router(settings))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": settings.ollama_model}

    return app


def main() -> None:
    """Execute either the realtime server or the Phase 2 microphone CLI."""
    logger = get_logger("main")
    session_id = os.getenv("SESSION_ID")
    settings = load_settings()

    if os.getenv("RUN_MODE", "server") == "cli":
        logger.info("Running Phase 2 live voice CLI pipeline")
        output = run_live_voice_pipeline(session_id=session_id)
        print(json.dumps(_model_to_dict(output), indent=2))
        return

    logger.info("Starting Phase 3 realtime WebSocket server")
    uvicorn.run(
        "app.main:create_app",
        host=settings.websocket_host,
        port=settings.websocket_port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
