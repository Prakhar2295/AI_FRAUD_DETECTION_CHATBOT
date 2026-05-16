"""Centralized application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the offline and live voice pipelines."""

    project_root: Path
    backend_root: Path
    ollama_endpoint: str
    ollama_model: str
    ollama_timeout_seconds: int
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    recording_duration_seconds: float
    audio_sample_rate: int
    audio_channels: int
    temp_audio_dir: Path
    low_risk_max_score: int
    medium_risk_max_score: int
    log_level: str


def load_settings() -> Settings:
    """Load settings from environment variables with local defaults."""
    backend_root = Path(__file__).resolve().parents[2]
    project_root = backend_root.parent
    temp_audio_dir = Path(
        os.getenv("TEMP_AUDIO_DIR", str(backend_root / "data" / "tmp"))
    )

    return Settings(
        project_root=project_root,
        backend_root=backend_root,
        ollama_endpoint=os.getenv(
            "OLLAMA_ENDPOINT",
            "http://localhost:11434/api/generate",
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        recording_duration_seconds=float(os.getenv("RECORDING_DURATION_SECONDS", "5")),
        audio_sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
        audio_channels=int(os.getenv("AUDIO_CHANNELS", "1")),
        temp_audio_dir=temp_audio_dir.expanduser().resolve(),
        low_risk_max_score=int(os.getenv("LOW_RISK_MAX_SCORE", "30")),
        medium_risk_max_score=int(os.getenv("MEDIUM_RISK_MAX_SCORE", "70")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

