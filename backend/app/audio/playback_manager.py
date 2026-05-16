"""Playback manager for async-safe TTS audio output."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from app.utils.logger import get_logger


@dataclass(frozen=True)
class PlaybackFrame:
    pcm_bytes: bytes
    sample_rate: int
    channels: int


class PlaybackManager:
    """Queue-based playback manager that runs audio chunks sequentially."""

    def __init__(self, output_device: str | None = None, max_queue: int = 16) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.output_device = output_device
        self.queue: asyncio.Queue[PlaybackFrame] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self.current_state: str = "idle"

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())
            self.logger.info("Playback manager started")

    async def stop(self) -> None:
        self._stop_event.set()
        sd.stop()
        if self._task is not None:
            await self._task
        self.current_state = "stopped"
        self.logger.info("Playback manager stopped")

    async def enqueue(self, frame: PlaybackFrame) -> None:
        await self.start()
        await self.queue.put(frame)
        self.current_state = "queued"
        self.logger.info(
            "Enqueued playback frame sample_rate=%s channels=%s queue_size=%s",
            frame.sample_rate,
            frame.channels,
            self.queue.qsize(),
        )

    async def _run(self) -> None:
        self.current_state = "playing"
        while not self._stop_event.is_set():
            try:
                frame = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self.queue.empty():
                    self.current_state = "idle"
                continue

            if self._stop_event.is_set():
                break

            self.current_state = "playing"
            try:
                await asyncio.to_thread(self._play_frame, frame)
                self.logger.info("Finished playback frame")
            except Exception as exc:
                self.current_state = "error"
                self.logger.error("Playback frame failed: %s", exc)
            finally:
                self.queue.task_done()

        self.current_state = "idle"

    def _play_frame(self, frame: PlaybackFrame) -> None:
        samples = np.frombuffer(frame.pcm_bytes, dtype=np.int16)
        if frame.channels > 1:
            samples = samples.reshape(-1, frame.channels)

        self.logger.info(
            "Playing audio frame sample_rate=%s channels=%s length=%s",
            frame.sample_rate,
            frame.channels,
            len(samples),
        )
        sd.play(samples, frame.sample_rate, device=self.output_device, blocking=True)
