"""Conversation analysis endpoints for transcript-based testing."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config.settings import load_settings
from app.models.api_models import ConversationAnalysisRequest
from app.models.response_models import WorkflowOutputResponse
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.graph.fraud_workflow import build_fraud_workflow
from app.graph.state import create_initial_state
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger("ConversationAPI")


@router.post("/api/v1/conversation/analyze", tags=["Conversation Intelligence"], response_model=WorkflowOutputResponse)
async def analyze_conversation(request: ConversationAnalysisRequest) -> Any:
    settings = load_settings()
    start = time.monotonic()

    logger.info("Conversation API request start | endpoint=/api/v1/conversation/analyze | method=POST | session_id=%s | transcript_len=%d", request.session_id, len(request.transcript or ""))

    try:
        llm = OllamaLLMService(
            model_name=settings.ollama_model,
            endpoint=settings.ollama_endpoint,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        memory = MemoryService()

        workflow = build_fraud_workflow(llm_service=llm, memory_service=memory, settings=settings)

        initial_state = create_initial_state(session_id=request.session_id, transcript=request.transcript)

        # execute synchronous workflow in a thread
        final_state = await asyncio.to_thread(workflow.invoke, initial_state)

        elapsed = time.monotonic() - start
        logger.info("Conversation analysis completed | session_id=%s | elapsed_s=%.3f | retrieved=%s", request.session_id, elapsed, len(final_state.get("retrieved_fraud_patterns") or []))
        return WorkflowOutputResponse(**final_state)
    except Exception as exc:
        logger.exception("Conversation analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
