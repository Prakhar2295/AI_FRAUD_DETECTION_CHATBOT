"""Abstract TTS service interface for streaming voice synthesis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from app.tts.voice_config import VoiceConfig


@dataclass(frozen=True)
class TTSAudioChunk:
    """A chunk of PCM16 LE audio returned by a TTS provider."""

    pcm_bytes: bytes
    sample_rate: int
    channels: int = 1


class TTSService:
    """Base class for TTS providers."""

    async def synthesize_text_to_pcm(self, text: str, voice_config: VoiceConfig) -> TTSAudioChunk:
        raise NotImplementedError

    async def synthesize_text_stream(
        self,
        text: str,
        voice_config: VoiceConfig,
        chunk_size: int = 4096,
    ) -> AsyncIterator[TTSAudioChunk]:
        audio = await self.synthesize_text_to_pcm(text, voice_config)
        for offset in range(0, len(audio.pcm_bytes), chunk_size):
            yield TTSAudioChunk(
                pcm_bytes=audio.pcm_bytes[offset : offset + chunk_size],
                sample_rate=audio.sample_rate,
                channels=audio.channels,
            )


class TTSConfigurationError(RuntimeError):
    """Raised when the TTS service cannot be configured for the current environment."""


class TTSGenerationError(RuntimeError):
    """Raised when text-to-speech synthesis fails."""
