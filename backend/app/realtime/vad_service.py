"""Lightweight voice activity detection abstraction."""

from __future__ import annotations

import numpy as np

from app.config.settings import Settings
from app.utils.logger import get_logger


class VADService:
    """Simple RMS-based speech detector prepared for future WebRTC VAD swap-in."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    def has_speech(self, audio_chunk: bytes) -> bool:
        """Return whether a PCM16 audio chunk appears to contain speech."""
        if not audio_chunk:
            return False

        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        if samples.size == 0:
            return False

        normalized = samples.astype(np.float32) / 32768.0
        normalized_energy = float(np.sqrt(np.mean(np.square(normalized))))
        has_speech = normalized_energy >= self.settings.vad_energy_threshold
        self.logger.info(
            "VAD chunk energy=%.4f speech=%s",
            normalized_energy,
            has_speech,
        )
        return has_speech
