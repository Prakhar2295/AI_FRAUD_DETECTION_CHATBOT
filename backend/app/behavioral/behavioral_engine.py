"""Orchestrate behavioral fraud intelligence analysis."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from .behavioral_pipeline import BehavioralPipeline


class BehavioralEngine:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.loop = loop or asyncio.get_event_loop()
        self.pipeline = BehavioralPipeline(loop=self.loop)

    async def analyze(self, transcript: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.pipeline.enqueue_transcript(
            session_id=metadata.get("session_id") if metadata else "unknown",
            transcript=transcript,
            metadata=metadata or {},
        )

    async def stop(self) -> None:
        await self.pipeline.stop()
