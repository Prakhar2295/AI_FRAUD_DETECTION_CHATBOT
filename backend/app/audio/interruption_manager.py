"""Interruption-aware playback control foundation."""

from __future__ import annotations

from app.audio.playback_manager import PlaybackManager
from app.utils.logger import get_logger


class InterruptionManager:
    """Manage playback interruption requests and safe stop behavior."""

    def __init__(self, playback_manager: PlaybackManager) -> None:
        self.playback_manager = playback_manager
        self.logger = get_logger(self.__class__.__name__)
        self.interruption_requested = False

    async def request_interrupt(self) -> None:
        self.interruption_requested = True
        self.logger.info("Interruption requested")
        await self.playback_manager.stop()

    def clear_interrupt(self) -> None:
        self.interruption_requested = False
        self.logger.info("Interruption cleared")

    def snapshot(self) -> dict[str, bool]:
        return {"interruption_requested": self.interruption_requested}
