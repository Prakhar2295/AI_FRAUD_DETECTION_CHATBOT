"""Audio output service for queued TTS playback."""

from __future__ import annotations

import asyncio
from typing import Any

from app.audio.playback_manager import PlaybackFrame, PlaybackManager
from app.utils.logger import get_logger


class AudioOutputService:
    """High-level audio playback service for TTS response output."""

    def __init__(self, output_device: str | None = None, max_queue: int = 16) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.playback_manager = PlaybackManager(output_device=output_device, max_queue=max_queue)
        self._status_lock = asyncio.Lock()

    async def enqueue_audio(self, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
        frame = PlaybackFrame(pcm_bytes=pcm_bytes, sample_rate=sample_rate, channels=channels)
        try:
            await self.playback_manager.enqueue(frame)
            self.logger.info("Audio enqueued for playback")
        except asyncio.QueueFull as exc:
            self.logger.error("Audio queue overflow: %s", exc)
            raise

    async def stop(self) -> None:
        await self.playback_manager.stop()

    async def status(self) -> dict[str, Any]:
        async with self._status_lock:
            return {
                "playback_state": self.playback_manager.current_state,
                "queue_size": self.playback_manager.queue.qsize(),
            }
