"""Live microphone capture service."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from app.config.settings import Settings
from app.utils.logger import get_logger


class MicrophoneService:
    """Capture microphone audio and persist it as a temporary WAV file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    def capture_audio(self) -> Path:
        """Record microphone audio and return the saved WAV file path."""
        self.settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        frames = int(
            self.settings.recording_duration_seconds * self.settings.audio_sample_rate
        )
        output_path = self._build_output_path()

        try:
            self.logger.info(
                "Recording microphone audio: duration=%ss sample_rate=%s channels=%s",
                self.settings.recording_duration_seconds,
                self.settings.audio_sample_rate,
                self.settings.audio_channels,
            )
            recording = sd.rec(
                frames,
                samplerate=self.settings.audio_sample_rate,
                channels=self.settings.audio_channels,
                dtype="float32",
            )
            sd.wait()
            wav_audio = self._to_int16(recording)
            write(output_path, self.settings.audio_sample_rate, wav_audio)
            self.logger.info("Saved microphone recording to %s", output_path)
            return output_path
        except Exception:
            self.logger.exception("Microphone capture failed")
            raise

    async def capture_audio_async(self) -> Path:
        """Async-ready wrapper for microphone capture."""
        return await asyncio.to_thread(self.capture_audio)

    def _build_output_path(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return self.settings.temp_audio_dir / f"mic_capture_{timestamp}.wav"

    @staticmethod
    def _to_int16(audio: np.ndarray) -> np.ndarray:
        """Convert float32 audio in [-1, 1] into int16 WAV samples."""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

