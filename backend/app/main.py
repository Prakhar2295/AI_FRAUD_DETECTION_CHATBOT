"""Phase 1 offline voice-to-intelligence pipeline entry point."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.fraud_analysis_service import FraudAnalysisService
from app.services.llm_service import OllamaLLMService
from app.services.stt_service import STTService
from app.utils.logger import get_logger


def _default_audio_path() -> Path:
    """Return the default Phase 1 sample audio path."""
    return BACKEND_ROOT / "data" / "fraud_detection_sample.wav"


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Support Pydantic v1 and v2 serialization."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def run_pipeline(audio_path: Path) -> dict[str, Any]:
    """Run STT followed by fraud analysis and return structured output."""
    stt_service = STTService(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    )
    llm_service = OllamaLLMService(
        model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
    )
    fraud_service = FraudAnalysisService(llm_service=llm_service)

    transcription = stt_service.transcribe_audio(audio_path)
    analysis = fraud_service.analyze_transcription(transcription.text)

    return {
        "transcription": _model_to_dict(transcription),
        "fraud_analysis": _model_to_dict(analysis),
    }


def main() -> None:
    """Execute the Phase 1 pipeline using the configured audio file."""
    logger = get_logger("main")
    audio_path = Path(os.getenv("AUDIO_FILE_PATH", str(_default_audio_path())))

    logger.info("Running Phase 1 pipeline")
    output = run_pipeline(audio_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

