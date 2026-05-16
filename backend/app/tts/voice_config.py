"""Voice configuration objects for TTS synthesis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceConfig:
    """Structured voice settings for TTS generation."""

    voice_name: str
    style: str
    language: str
    sample_rate: int
    channels: int = 1
