"""Async queue manager for realtime audio chunks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.config.settings import Settings
from app.models.websocket_models import AudioChunkMetadata
from app.utils.logger import get_logger


@dataclass(frozen=True)
class AudioQueueItem:
    """Item placed on the streaming audio queue."""

    payload: bytes | None
    metadata: AudioChunkMetadata | None
    kind: str = "audio"


class AudioStreamManager:
    """Session-scoped asyncio queue for inbound audio chunks."""

    def __init__(self, settings: Settings, session_id: str) -> None:
        self.settings = settings
        self.session_id = session_id
        self.queue: asyncio.Queue[AudioQueueItem] = asyncio.Queue(
            maxsize=settings.audio_queue_max_size
        )
        self.logger = get_logger(f"{self.__class__.__name__}.{session_id}")

    async def enqueue_audio(
        self,
        payload: bytes,
        metadata: AudioChunkMetadata,
    ) -> None:
        """Put an audio chunk on the queue or raise when the queue is full."""
        if self.queue.full():
            raise asyncio.QueueFull("Audio queue is full")
        await self.queue.put(AudioQueueItem(payload=payload, metadata=metadata))
        self.logger.info("Audio queued: sequence=%s size=%s", metadata.sequence, self.queue.qsize())

    async def enqueue_flush(self) -> None:
        """Request processing of the current buffered audio window."""
        await self.queue.put(AudioQueueItem(payload=None, metadata=None, kind="flush"))

    async def enqueue_stop(self) -> None:
        """Signal the consumer to stop."""
        await self.queue.put(AudioQueueItem(payload=None, metadata=None, kind="stop"))

    async def next_item(self) -> AudioQueueItem:
        """Read the next queued item."""
        return await self.queue.get()

