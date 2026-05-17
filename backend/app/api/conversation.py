"""Conversation analysis endpoints for transcript-based testing."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config.settings import load_settings
from app.models.api_models import ConversationAnalysisRequest
from app.models.response_models import WorkflowOutputResponse, RiskAnalysisResponse
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

        print("Final state from workflow execution:", final_state)  # Debug log to inspect final state structure

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

        # Normalize final state into WorkflowOutputResponse fields (provide defaults)
        try:
            risk = RiskAnalysisResponse(**_state_to_dict(final.get("risk", {})))
        except Exception:
            # If risk is missing or malformed, set safe defaults
            logger.exception("Risk parsing failed; using defaults")
            risk = RiskAnalysisResponse(fraud_risk_score=0, risk_level="low", reasoning_summary="")

        history = final.get("conversation_history", []) or []

        response = WorkflowOutputResponse(
            session_id=final.get("session_id", request.session_id),
            transcript=final.get("transcript", request.transcript),
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

        elapsed = time.monotonic() - start
        logger.info(
            "Conversation analysis completed | session_id=%s | elapsed_s=%.3f | retrieved=%s",
            request.session_id,
            elapsed,
            len(response.retrieved_fraud_patterns or []),
        )
        # Return a plain dict to avoid any model re-validation issues in the
        # FastAPI response pipeline and ensure all required fields are present.
        try:
            return response.model_dump()
        except Exception:
            logger.exception("Failed to dump WorkflowOutputResponse model; returning safe dict")
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
    except Exception as exc:
        logger.exception("Conversation analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
