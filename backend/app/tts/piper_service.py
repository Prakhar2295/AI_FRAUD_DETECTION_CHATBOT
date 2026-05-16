"""Piper TTS provider integration."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.tts.tts_service import TTSGenerationError, TTSService, TTSAudioChunk
from app.tts.voice_config import VoiceConfig
from app.utils.logger import get_logger


@dataclass(frozen=True)
class PiperTTSConfiguration:
    model_name: str
    voice_name: str
    sample_rate: int
    channels: int = 1
    piper_binary: str = "piper"


class PiperTTSService(TTSService):
    """Local Piper TTS integration for Apple Silicon and future voice cloning."""

    def __init__(self, config: PiperTTSConfiguration) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    async def synthesize_text_to_pcm(self, text: str, voice_config: VoiceConfig) -> TTSAudioChunk:
        output_path = Path("/tmp") / f"tts_response_{hash(text) & 0xFFFFFFFF:x}.wav"
        command = [
            self.config.piper_binary,
            "speak",
            "--model",
            self.config.model_name,
            "--voice",
            self.config.voice_name,
            "--text",
            text,
            "--output",
            str(output_path),
            "--sample-rate",
            str(self.config.sample_rate),
        ]

        self.logger.info("Running Piper TTS command: %s", " ".join(command))

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise TTSGenerationError(
                    f"Piper exited {process.returncode}: {stderr.decode().strip()}"
                )
        except FileNotFoundError as exc:
            raise TTSGenerationError(
                "Piper binary not found. Install Piper and ensure it is on PATH."
            ) from exc
        except Exception as exc:
            raise TTSGenerationError("Piper synthesis failed") from exc

        if not output_path.exists():
            raise TTSGenerationError("Piper did not produce an output file")

        try:
            from scipy.io.wavfile import read

            sample_rate, data = read(output_path)
            if data.dtype != "int16":
                raise TTSGenerationError("Unexpected Piper output sample type")
            bytes_data = data.tobytes()
            return TTSAudioChunk(
                pcm_bytes=bytes_data,
                sample_rate=sample_rate,
                channels=data.shape[1] if data.ndim > 1 else 1,
            )
        except Exception as exc:
            raise TTSGenerationError("Failed to read Piper output") from exc


class PiperNotInstalledTTSService(TTSService):
    """Fallback TTS service for environments without Piper installed."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger = get_logger(self.__class__.__name__)

    async def synthesize_text_to_pcm(self, text: str, voice_config: VoiceConfig) -> TTSAudioChunk:
        self.logger.warning("Piper not available; generating silent placeholder audio")
        duration_seconds = min(max(len(text) / 20.0, 0.5), 3.0)
        frame_count = int(duration_seconds * self.sample_rate)
        silence = b"\x00\x00" * frame_count * self.channels
        return TTSAudioChunk(
            pcm_bytes=silence,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
