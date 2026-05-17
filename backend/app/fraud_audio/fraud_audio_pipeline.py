"""Async-safe fraud audio pipeline coordinator.

This coordinator accepts streaming PCM frames and schedules authenticity
analysis without blocking the realtime websocket handlers.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Any
import numpy as np

from .authenticity_engine import AuthenticityEngine


class FraudAudioPipeline:
    def __init__(self, *, max_queue: int = 64, loop: asyncio.AbstractEventLoop | None = None):
        self.loop = loop or asyncio.get_event_loop()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.engine = AuthenticityEngine(self.loop)
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = self.loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue_audio(self, session_id: str, pcm: bytes, sr: int, timestamp: float) -> asyncio.Future:
        """Enqueue raw PCM bytes for asynchronous analysis. Returns a Future that resolves to the result dict."""
        loop = self.loop
        fut: asyncio.Future = loop.create_future()
        try:
            pcm_arr = np.frombuffer(pcm, dtype=np.int16).astype(float)
        except Exception as e:
            fut.set_exception(e)
            return fut

        item = {"session_id": session_id, "pcm": pcm_arr, "sr": sr, "timestamp": timestamp, "future": fut}
        await self.queue.put(item)
        return fut

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            fut: asyncio.Future = item.get("future")
            try:
                result = await self.engine.analyze(item["pcm"], item["sr"])
                if fut and not fut.done():
                    fut.set_result(result)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self.queue.task_done()
