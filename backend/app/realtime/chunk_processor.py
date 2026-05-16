"""Chunk normalization and aggregation for pseudo-streaming STT."""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import Settings
from app.models.websocket_models import AudioChunkMetadata
from app.realtime.vad_service import VADService
from app.utils.logger import get_logger


@dataclass(frozen=True)
class ProcessedAudioChunk:
    """Validated audio chunk plus metadata."""

    payload: bytes
    metadata: AudioChunkMetadata
    has_speech: bool


@dataclass(frozen=True)
class TranscriptionWindow:
    """Aggregated audio window ready for transcription."""

    payload: bytes
    start_sequence: int
    end_sequence: int
    sample_rate: int
    channels: int


class ChunkProcessor:
    """Validate, normalize, and aggregate incoming audio chunks."""

    def __init__(self, settings: Settings, vad_service: VADService) -> None:
        self.settings = settings
        self.vad_service = vad_service
        self.logger = get_logger(self.__class__.__name__)
        self._buffer: list[ProcessedAudioChunk] = []
        self._buffered_bytes = 0
        self._speech_chunks = 0

    def process_chunk(
        self,
        payload: bytes,
        metadata: AudioChunkMetadata,
    ) -> TranscriptionWindow | None:
        """Process one chunk and return a transcription window when ready."""
        self._validate_chunk(payload, metadata)
        has_speech = self.vad_service.has_speech(payload)
        chunk = ProcessedAudioChunk(
            payload=payload,
            metadata=metadata,
            has_speech=has_speech,
        )

        self._buffer.append(chunk)
        self._buffered_bytes += len(payload)
        self._speech_chunks += int(has_speech)
        self.logger.info(
            "Buffered audio chunk: sequence=%s buffered_bytes=%s",
            metadata.sequence,
            self._buffered_bytes,
        )

        if self._is_window_ready():
            return self.flush()
        return None

    def flush(self) -> TranscriptionWindow | None:
        """Flush the current buffer into a transcription window."""
        if not self._buffer:
            return None

        start_sequence = self._buffer[0].metadata.sequence
        end_sequence = self._buffer[-1].metadata.sequence
        sample_rate = self._buffer[0].metadata.sample_rate
        channels = self._buffer[0].metadata.channels
        payload = b"".join(chunk.payload for chunk in self._buffer)
        speech_chunks = self._speech_chunks

        self._buffer = []
        self._buffered_bytes = 0
        self._speech_chunks = 0

        if speech_chunks < self.settings.vad_min_speech_chunks:
            self.logger.info("Skipping silent transcription window")
            return None

        return TranscriptionWindow(
            payload=payload,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            sample_rate=sample_rate,
            channels=channels,
        )

    def _validate_chunk(self, payload: bytes, metadata: AudioChunkMetadata) -> None:
        if not payload:
            raise ValueError("Audio chunk payload cannot be empty")
        if metadata.sample_rate != self.settings.audio_sample_rate:
            raise ValueError(
                f"Unsupported sample rate {metadata.sample_rate}; "
                f"expected {self.settings.audio_sample_rate}"
            )
        if metadata.channels != self.settings.audio_channels:
            raise ValueError(
                f"Unsupported channel count {metadata.channels}; "
                f"expected {self.settings.audio_channels}"
            )
        if metadata.encoding != "pcm_s16le":
            raise ValueError(f"Unsupported streaming encoding: {metadata.encoding}")

    def _is_window_ready(self) -> bool:
        bytes_per_second = self.settings.audio_sample_rate
        bytes_per_second *= self.settings.audio_channels
        bytes_per_second *= 2
        target_bytes = int(bytes_per_second * self.settings.audio_window_seconds)
        return self._buffered_bytes >= target_bytes

