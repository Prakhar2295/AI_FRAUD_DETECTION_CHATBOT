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
from app.graph.fraud_workflow import build_fraud_workflow
from app.graph.state import create_initial_state
from app.models.response_models import WorkflowOutputResponse
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

        # Stage 4: LangGraph workflow
        logger.info("LangGraph workflow initialization | session_id=%s", session_id)
        workflow = build_fraud_workflow(llm_service=llm, memory_service=memory, settings=settings)

        initial_state = create_initial_state(session_id=session_id, transcript=transcription.text)
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

        return WorkflowOutputResponse(**final_state)
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
