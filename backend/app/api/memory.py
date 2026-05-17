"""Fraud memory inspection endpoints."""
from __future__ import annotations

from typing import Any

import asyncio
from fastapi import APIRouter, HTTPException

from app.services.memory_service import MemoryService
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger("MemoryAPI")


@router.get("/api/v1/memory/session/{session_id}", tags=["Fraud Memory"])
async def get_session_memory(session_id: str) -> Any:
    logger.info("Memory API request start | endpoint=/api/v1/memory/session/{session_id} | method=GET | session_id=%s", session_id)
    try:
        mem = MemoryService()
        session = await __async_get_session(mem, session_id)
        # normalize into lightweight JSON
        history = [
            {
                "timestamp": i.timestamp,
                "transcript": i.transcript,
                "risk_level": getattr(i.risk, "risk_level", None) if i.risk else None,
            }
            for i in session.interactions
        ]
        logger.info("Memory API request completed | session_id=%s | interactions=%d", session.session_id, len(history))
        return {"session_id": session.session_id, "interaction_count": len(history), "history": history}
    except Exception as exc:
        logger.exception("Failed to fetch session memory: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


async def __async_get_session(mem: MemoryService, session_id: str):
    return await asyncio.to_thread(mem.get_session, session_id)
