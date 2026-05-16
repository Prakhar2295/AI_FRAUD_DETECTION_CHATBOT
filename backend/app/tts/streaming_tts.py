"""Streaming TTS chunk wrapper."""

from __future__ import annotations

from typing import AsyncIterator

from app.tts.tts_service import TTSService, TTSAudioChunk
from app.tts.voice_config import VoiceConfig


class StreamingTTS:
    """Wrap a TTS provider with a chunked streaming interface."""

    def __init__(self, tts_service: TTSService, chunk_size: int = 4096) -> None:
        self.tts_service = tts_service
        self.chunk_size = chunk_size

    async def stream_text(self, text: str, voice_config: VoiceConfig) -> AsyncIterator[TTSAudioChunk]:
        audio = await self.tts_service.synthesize_text_to_pcm(text, voice_config)
        for offset in range(0, len(audio.pcm_bytes), self.chunk_size):
            yield TTSAudioChunk(
                pcm_bytes=audio.pcm_bytes[offset : offset + self.chunk_size],
                sample_rate=audio.sample_rate,
                channels=audio.channels,
            )
