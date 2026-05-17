"""Health check endpoints for system readiness."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

import requests
from fastapi import APIRouter

from app.services.llm_service import OllamaLLMService
from app.services.stt_service import STTService
from app.services.memory_service import MemoryService
from app.vectorstore.chroma_manager import ChromaManager
from app.config.settings import load_settings
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger("HealthAPI")


@router.get("/health", tags=["Health"])
async def health() -> Dict[str, Any]:
    settings = load_settings()
    logger.info("Health API request start | endpoint=/health | method=GET")

    async def _check_ollama() -> str:
        try:
            svc = OllamaLLMService(model_name=settings.ollama_model, endpoint=settings.ollama_endpoint)
            # quick connectivity check: attempt a short HEAD/GET
            resp = await asyncio.to_thread(requests.get, svc.endpoint, timeout=2)
            return "connected" if resp is not None else "unreachable"
        except Exception:
            logger.exception("Ollama health check failed")
            return "unreachable"

    async def _check_whisper() -> str:
        try:
            stt = STTService(
                model_size=settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            # attempt to touch the cached property with a short timeout
            try:
                await asyncio.wait_for(asyncio.to_thread(lambda: getattr(stt, "model")), timeout=5)
                return "loaded"
            except asyncio.TimeoutError:
                return "deferred"
        except Exception:
            logger.exception("Whisper health check failed")
            return "unavailable"

    async def _check_chroma() -> str:
        try:
            cm = ChromaManager()
            return "connected" if getattr(cm, "collection", None) is not None else "fallback"
        except Exception:
            logger.exception("Chroma health check failed")
            return "unavailable"

    async def _check_memory() -> str:
        try:
            mem = MemoryService()
            # call get_session to ensure memory layer responds
            await asyncio.to_thread(mem.get_session, "health-check")
            return "ready"
        except Exception:
            logger.exception("Memory health check failed")
            return "unavailable"

    ollama, whisper, chroma, memory = await asyncio.gather(
        _check_ollama(), _check_whisper(), _check_chroma(), _check_memory()
    )

    logger.info("Health API request completed | services=%s", {"ollama": ollama, "whisper": whisper, "chromadb": chroma, "memory": memory})

    return {
        "status": "healthy",
        "services": {
            "ollama": ollama,
            "whisper": whisper,
            "chromadb": chroma,
            "memory": memory,
        },
        "model": settings.ollama_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
