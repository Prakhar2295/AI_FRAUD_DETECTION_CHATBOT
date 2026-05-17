"""Async queue for persistent fraud memory ingestion."""
from __future__ import annotations

import asyncio
from typing import Any

from app.utils.logger import get_logger


class MemoryPipeline:
    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None, max_queue: int = 128):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self.logger = get_logger(self.__class__.__name__)

    async def start(self) -> None:
        self.loop = self.loop or asyncio.get_running_loop()
        if self._task is None or self._task.done():
            self._task = self.loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)
        await self.start()

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                task = item.get("task")
                handler = item.get("handler")
                if callable(handler):
                    await handler()
            except Exception as exc:
                self.logger.exception("MemoryPipeline task failed: %s", exc)
            finally:
                self.queue.task_done()
