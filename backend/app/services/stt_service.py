"""Speech-to-text service backed by faster-whisper."""

from __future__ import annotations

import asyncio
from functools import cached_property
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from app.models.response_models import TranscriptionResponse
from app.utils.logger import get_logger


class STTService:
    """Reusable, lazy-loading Whisper transcription service."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.logger = get_logger(self.__class__.__name__)

    @cached_property
    def model(self) -> WhisperModel:
        """Load the Whisper model once per service instance."""
        self.logger.info(
            "Loading Whisper model: size=%s device=%s compute_type=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe_audio(self, audio_path: Path) -> TranscriptionResponse:
        """Transcribe a WAV audio file and return clean text."""
        resolved_path = audio_path.expanduser().resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Audio file not found: {resolved_path}")

        if resolved_path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a .wav file, received: {resolved_path.suffix}")

        try:
            self.logger.info("Starting transcription for %s", resolved_path)
            segments, info = self.model.transcribe(str(resolved_path), vad_filter=True)
            text = self._join_segments(segments)

            self.logger.info("Transcription completed: %.0f characters", len(text))
            return TranscriptionResponse(
                audio_path=str(resolved_path),
                text=text,
                language=info.language,
                duration_seconds=self._duration_seconds(info),
            )
        except Exception:
            self.logger.exception("Transcription failed for %s", resolved_path)
            raise

    async def transcribe_audio_async(self, audio_path: Path) -> TranscriptionResponse:
        """Async-ready wrapper for file transcription."""
        return await asyncio.to_thread(self.transcribe_audio, audio_path)

    @staticmethod
    def _join_segments(segments: Iterable[Segment]) -> str:
        """Combine Whisper segments into a normalized transcription string."""
        return " ".join(segment.text.strip() for segment in segments).strip()

    @staticmethod
    def _duration_seconds(info: TranscriptionInfo) -> float | None:
        """Return audio duration when available from faster-whisper."""
        duration = getattr(info, "duration", None)
        return float(duration) if duration is not None else None
