# Banking Fraud Detection Voice AI - Phase 1

Phase 1 built an offline voice intelligence pipeline:

```text
Audio File (.wav)
-> faster-whisper speech-to-text
-> local Ollama LLM
-> banking fraud reasoning
-> structured AI response
```

This phase intentionally did not include realtime streaming, live microphone capture, WebSockets, Docker, LangGraph, or a frontend.

## Architecture

```text
backend/
├── app/
│   ├── services/
│   │   ├── stt_service.py
│   │   ├── llm_service.py
│   │   └── fraud_analysis_service.py
│   ├── models/
│   │   └── response_models.py
│   ├── utils/
│   │   └── logger.py
│   └── main.py
├── data/
│   └── fraud_detection_sample.wav
├── requirements.txt
└── README_Phase_1.md
```

## Setup

From the project root:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Make sure Ollama is running and the model is available:

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

## Phase 1 Run Path

Phase 1 used a sample WAV file at:

```text
backend/data/fraud_detection_sample.wav
```

The Phase 1 flow was:

```text
sample WAV -> STTService -> FraudAnalysisService -> OllamaLLMService -> structured JSON
```

The current `main.py` now runs the Phase 2 microphone workflow, but the Phase 1 services remain available and reusable.

## Service Responsibilities

- `STTService`: owns faster-whisper model loading and WAV transcription.
- `OllamaLLMService`: owns local Ollama API calls, timeout handling, and JSON parsing.
- `FraudAnalysisService`: owns fraud prompt construction, LLM response validation, and risk schema generation.
- `response_models.py`: defines stable Pydantic contracts for downstream phases.
- `logger.py`: provides timestamped service-level logging.

## Phase 1 Output

The structured response included:

- transcription
- fraud risk score
- risk level
- suspicious indicators
- LLM reasoning

## Phase 1 Scalability Notes

Phase 1 established service boundaries that Phase 2 reused:

- STT remained isolated in `STTService`.
- Ollama calls remained isolated in `OllamaLLMService`.
- Fraud reasoning remained schema-driven through Pydantic models.
- The pipeline stayed local-first for Apple Silicon development.

