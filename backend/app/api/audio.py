"""Audio upload and end-to-end audio analysis endpoints."""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config.settings import load_settings
from app.services.stt_service import STTService
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.services.audio_emotion_service import AudioEmotionService
from app.graph.fraud_workflow import build_fraud_workflow
from app.graph.state import create_initial_state
from app.models.response_models import WorkflowOutputResponse, RiskAnalysisResponse
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger("AudioAPI")


def _validate_wav(upload: UploadFile) -> None:
    filename = upload.filename or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    content_type = (upload.content_type or "").lower()
    if content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream", "audio/wave"):
        # allow octet-stream for some clients
        logger.warning("Unexpected content_type for upload: %s", content_type)


@router.post("/api/v1/audio/analyze", tags=["Audio Intelligence"], response_model=WorkflowOutputResponse)
async def analyze_audio(file: UploadFile = File(...), session_id: str = Form(...), source: str | None = Form(None)) -> Any:
    settings = load_settings()
    upload_start = time.monotonic()
    tmp_path = None

    logger.info("Audio API request start | endpoint=/api/v1/audio/analyze | method=POST | session_id=%s | filename=%s", session_id, getattr(file, "filename", None))

    try:
        # Stage 1: validation
        logger.info("Audio upload received | session_id=%s filename=%s", session_id, getattr(file, "filename", None))
        _validate_wav(file)
        logger.info("Audio validation completed | session_id=%s", session_id)

        # Stage 2: write temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="upload_") as tmp:
            tmp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)
        upload_latency = (time.monotonic() - upload_start) * 1000.0
        logger.info("Temporary file created | path=%s | session_id=%s | upload_ms=%.1f", tmp_path, session_id, upload_latency)

        stt = STTService(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

        # Stage 3: STT
        logger.info("STT transcription started | session_id=%s | path=%s", session_id, tmp_path)
        stt_start = time.monotonic()
        transcription = await stt.transcribe_audio_async(tmp_path)
        stt_latency = (time.monotonic() - stt_start) * 1000.0
        logger.info("STT transcription completed | session_id=%s | stt_ms=%.1f | chars=%d", session_id, stt_latency, len(transcription.text or ""))
        logger.debug("Transcript generated | session_id=%s | transcript=%s", session_id, transcription.text or "")

        llm = OllamaLLMService(
            model_name=settings.ollama_model,
            endpoint=settings.ollama_endpoint,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        memory = MemoryService()

        # Stage 3.5: SER - Speech Emotion Recognition
        try:
            aes = AudioEmotionService()
            ser_start = time.monotonic()
            audio_emotion = await asyncio.to_thread(aes.analyze_file, str(tmp_path))
            ser_latency = (time.monotonic() - ser_start) * 1000.0
            logger.info("SER completed | session_id=%s | ser_ms=%.1f | dominant=%s", session_id, ser_latency, audio_emotion.get("dominant_emotion"))
            logger.debug("SER output | session_id=%s | output=%s", session_id, audio_emotion)
        except Exception:
            logger.exception("SER analysis failed; continuing without audio emotion")
            audio_emotion = None

        # Stage 4: LangGraph workflow
        logger.info("LangGraph workflow initialization | session_id=%s", session_id)
        workflow = build_fraud_workflow(llm_service=llm, memory_service=memory, settings=settings)

        initial_state = create_initial_state(session_id=session_id, transcript=transcription.text, audio_emotion_analysis=audio_emotion)
        workflow_start = time.monotonic()
        final_state = await asyncio.to_thread(workflow.invoke, initial_state)
        workflow_latency = (time.monotonic() - workflow_start) * 1000.0

        # Log key workflow outputs for visibility
        retrieved = final_state.get("retrieved_fraud_patterns") or []
        retrieval_meta = final_state.get("semantic_retrieval_metadata") or {}
        adaptive = final_state.get("adaptive_risk_enrichment") or {}
        fraud_context = final_state.get("fraud_knowledge_context") or {}
        behavioral = final_state.get("behavioral")

        logger.info(
            "Workflow completed | session_id=%s | workflow_ms=%.1f | total_ms=%.1f | retrieved=%s | retrieval_meta=%s | adaptive=%s",
            session_id,
            workflow_latency,
            (time.monotonic() - upload_start) * 1000.0,
            len(retrieved),
            retrieval_meta,
            adaptive,
        )

        if behavioral:
            logger.info("Behavioral analysis summary | session_id=%s | behavioral=%s", session_id, behavioral)

        # Ensure final_state is a plain dict (workflow may return pydantic models)
        def _state_to_dict(s: object) -> dict:
            if s is None:
                return {}
            if isinstance(s, dict):
                return s
            if hasattr(s, "model_dump"):
                try:
                    return s.model_dump()
                except Exception:
                    pass
            if hasattr(s, "dict"):
                try:
                    return s.dict()
                except Exception:
                    pass
            try:
                return dict(s)
            except Exception:
                return {}

        final = _state_to_dict(final_state)

        # Normalize risk output and provide safe defaults if parsing fails
        try:
            risk = RiskAnalysisResponse(**_state_to_dict(final.get("risk", {})))
        except Exception:
            logger.exception("Risk parsing failed in audio API; using defaults")
            risk = RiskAnalysisResponse(fraud_risk_score=0, risk_level="low", reasoning_summary="")

        history = final.get("conversation_history", []) or []

        response = WorkflowOutputResponse(
            session_id=final.get("session_id", session_id),
            transcript=final.get("transcript", transcription.text),
            partial_transcript=final.get("partial_transcript"),
            stream_sequence=final.get("stream_sequence"),
            intent_classification=final.get("intent"),
            suspicious_indicators=final.get("suspicious_indicators", []),
            fraud_risk_score=risk.fraud_risk_score,
            risk_level=risk.risk_level,
            reasoning_summary=risk.reasoning_summary,
            workflow_execution_trace=final.get("workflow_trace", []),
            node_execution_timestamps=final.get("node_timestamps", {}),
            conversation_turn_count=len(history),
            retrieved_fraud_patterns=final.get("retrieved_fraud_patterns", []),
            semantic_retrieval_metadata=final.get("semantic_retrieval_metadata", {}),
            historical_context=final.get("historical_fraud_context"),
            adaptive_risk_enrichment=final.get("adaptive_risk_enrichment", {}),
            fraud_knowledge_context=final.get("fraud_knowledge_context"),
            behavioral=final.get("behavioral"),
            ai_response=final.get("ai_response"),
            errors=final.get("errors", []),
        )

        # Return a plain dict to avoid model re-validation issues in FastAPI
        try:
            return response.model_dump()
        except Exception:
            logger.exception("Failed to dump WorkflowOutputResponse model in audio API; returning safe dict")
            return {
                "session_id": response.session_id,
                "transcript": response.transcript,
                "partial_transcript": response.partial_transcript,
                "stream_sequence": response.stream_sequence,
                "intent_classification": None,
                "suspicious_indicators": response.suspicious_indicators or [],
                "fraud_risk_score": response.fraud_risk_score,
                "risk_level": response.risk_level,
                "reasoning_summary": response.reasoning_summary,
                "workflow_execution_trace": response.workflow_execution_trace or [],
                "node_execution_timestamps": response.node_execution_timestamps or {},
                "conversation_turn_count": response.conversation_turn_count or 0,
                "retrieved_fraud_patterns": response.retrieved_fraud_patterns or [],
                "semantic_retrieval_metadata": response.semantic_retrieval_metadata or {},
                "historical_context": response.historical_context,
                "adaptive_risk_enrichment": response.adaptive_risk_enrichment or {},
                "fraud_knowledge_context": response.fraud_knowledge_context,
                "behavioral": None,
                "ai_response": None,
                "errors": response.errors or [],
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Audio analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and tmp_path.exists():
            try:
                os.remove(tmp_path)
                logger.info("Cleaned up uploaded file %s", tmp_path)
            except Exception:
                logger.exception("Failed to delete temp file %s", tmp_path)
