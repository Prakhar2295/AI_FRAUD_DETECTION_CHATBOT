"""Centralized application configuration using pydantic BaseSettings.

Uses a .env file at the repository root and environment variables. Call
`load_settings()` to get a cached Settings instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App roots
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2].parent)
    backend_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    # LLM / Ollama
    ollama_endpoint: str = Field(default="http://localhost:11434/api/generate")
    ollama_model: str = Field(default="qwen2.5:1.5b")
    ollama_timeout_seconds: int = Field(default=120)

    # Embeddings
    embedding_provider: str = Field(default="sentence-transformers")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimensions: int = Field(default=384)
    max_embedding_text_length: int = Field(default=1500)

    # Vector retrieval
    vector_top_k: int = Field(default=4)
    vector_similarity_threshold: float = Field(default=0.55)
    enable_reranking: bool = Field(default=False)
    reranker_model: Optional[str] = Field(default=None)

    # SER model
    ser_model: str = Field(default="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")

    # Whisper / STT
    whisper_model_size: str = Field(default="base")
    whisper_device: str = Field(default="cpu")
    whisper_compute_type: str = Field(default="int8")

    # Retrieval tuning
    adaptive_retrieval_boost: float = Field(default=1.0)
    max_retrieval_boost: float = Field(default=2.0)
    max_history_penalty: float = Field(default=0.5)

    # Audio
    recording_duration_seconds: float = Field(default=5.0)
    audio_sample_rate: int = Field(default=16000)
    audio_channels: int = Field(default=1)
    temp_audio_dir: Path = Field(default_factory=lambda: Path("./data/tmp"))

    # Websocket
    websocket_host: str = Field(default="127.0.0.1")
    websocket_port: int = Field(default=8000)

    # TTS
    tts_voice_name: str = Field(default="alloy")
    tts_voice_style: str = Field(default="neutral")
    tts_language: str = Field(default="en")
    tts_output_device: Optional[str] = Field(default=None)

    # Chroma / Vector store
    chroma_persist_dir: Path = Field(default_factory=lambda: Path("./data/chroma"))
    chroma_collection_name: str = Field(default="fraud_memory")

    # VAD
    vad_energy_threshold: float = Field(default=0.01)
    vad_min_speech_chunks: int = Field(default=1)

    # Risk thresholds
    low_risk_max_score: int = Field(default=30)
    medium_risk_max_score: int = Field(default=70)

    # Logging
    log_level: str = Field(default="INFO")

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"


_CACHED_SETTINGS: Optional[Settings] = None


def load_settings() -> Settings:
    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is None:
        _CACHED_SETTINGS = Settings()
    return _CACHED_SETTINGS
