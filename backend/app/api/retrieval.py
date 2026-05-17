"""Semantic retrieval endpoints to validate vector search and memory.
"""
from __future__ import annotations

import time
from typing import Any, List

import asyncio
from fastapi import APIRouter, HTTPException

from app.config.settings import load_settings
from app.models.api_models import RetrievalSearchRequest, RetrievalResponse, RetrievalMatch
from app.services.memory_service import MemoryService
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger("RetrievalAPI")


@router.post("/api/v1/retrieval/search", tags=["Semantic Retrieval"], response_model=RetrievalResponse)
async def search_retrieval(request: RetrievalSearchRequest) -> Any:
    settings = load_settings()
    mem = MemoryService()
    start = time.monotonic()

    logger.info("Retrieval API request start | endpoint=/api/v1/retrieval/search | method=POST | session_id=%s | query_len=%d", request.session_id, len(request.query or ""))

    try:
        # MemoryService API expects (session_id, transcript)
        session_id = request.session_id or "global"
        retrieved = await __async_retrieve(mem, session_id, request.query)

        matches: List[RetrievalMatch] = []
        for hit in retrieved.get("hits", []):
            score = float(hit.get("similarity", 0.0))
            if score < (request.similarity_threshold or 0.0):
                continue
            matches.append(
                RetrievalMatch(
                    similarity_score=score,
                    transcript=hit.get("document", hit.get("content", "")),
                    risk_level=hit.get("metadata", {}).get("risk_level"),
                    fraud_risk_score=hit.get("metadata", {}).get("fraud_risk_score"),
                    metadata=hit.get("metadata", {}),
                )
            )

        metadata = {
            "vector_hits": len(matches),
            "threshold": request.similarity_threshold or 0.0,
            "elapsed_ms": (time.monotonic() - start) * 1000.0,
        }

        return RetrievalResponse(query=request.query, matches=matches, retrieval_metadata=metadata)
    except Exception as exc:
        logger.exception("Retrieval search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


async def __async_retrieve(mem: MemoryService, session_id: str, transcript: str) -> dict:
    return await asyncio.to_thread(mem.retrieve_similar_fraud, session_id, transcript)
