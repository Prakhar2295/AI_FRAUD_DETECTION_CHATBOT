"""Lightweight analytics endpoints for backend observability."""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

from app.memory.session_memory import SessionMemory
from app.vectorstore.chroma_manager import ChromaManager
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger("AnalyticsAPI")


@router.get("/api/v1/analytics/system", tags=["Analytics"])
async def system_metrics() -> Dict[str, Any]:
    logger.info("Analytics API request start | endpoint=/api/v1/analytics/system | method=GET")
    start = time.monotonic()
    session_mem = SessionMemory()
    chroma = ChromaManager()

    # best-effort metrics
    active_sessions = len(getattr(session_mem, "_sessions", {}))
    chroma_stats = {
        "collection": getattr(chroma, "collection", None) is not None,
        "persist_dir": str(getattr(chroma, "persist_dir", "")),
    }

    metrics = {
        "active_sessions": active_sessions,
        "average_stt_latency_ms": 0.0,
        "average_workflow_latency_ms": 0.0,
        "average_retrieval_latency_ms": 0.0,
        "memory_usage_bytes": 0,
        "chroma_stats": chroma_stats,
        "computed_at_ms": (time.monotonic() - start) * 1000.0,
    }
    logger.info("Reported analytics metrics: %s", metrics)
    return metrics
