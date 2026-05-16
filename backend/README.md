# Realtime AI Banking Fraud Detection Voice Agent - Phase 1

Phase 1 builds an offline voice intelligence pipeline:

```text
Audio File (.wav)
-> faster-whisper speech-to-text
-> local Ollama LLM
-> banking fraud reasoning
-> structured AI response
```

This phase intentionally does not include realtime streaming, live microphone capture, WebSockets, Docker, LangGraph, or a frontend.

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
└── README.md
```

## Setup

From the project root:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Make sure Ollama is running and the model is available:

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

## Run

```bash
python backend/app/main.py
```

The pipeline reads `backend/data/fraud_detection_sample.wav`, transcribes it, sends the transcription to local Ollama, and prints structured JSON.

## Configuration

Optional environment variables:

```bash
export AUDIO_FILE_PATH=/path/to/audio.wav
export WHISPER_MODEL_SIZE=base
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
export OLLAMA_MODEL=qwen2.5:1.5b
export OLLAMA_ENDPOINT=http://localhost:11434/api/generate
export OLLAMA_TIMEOUT_SECONDS=120
```

## Service Responsibilities

- `STTService`: owns faster-whisper model loading and WAV transcription.
- `OllamaLLMService`: owns local Ollama API calls, timeout handling, and JSON parsing.
- `FraudAnalysisService`: owns fraud prompt construction, LLM response validation, and risk schema generation.
- `response_models.py`: defines stable Pydantic contracts for downstream API and realtime phases.
- `logger.py`: provides timestamped service-level logging.

## Future Scalability Path

This structure is ready for the next phases:

- Add FastAPI routes without changing the core services.
- Add WebSocket streaming by feeding partial transcripts into `FraudAnalysisService`.
- Add live microphone capture as a new input adapter beside file-based STT.
- Expand fraud reasoning with rules, retrieval, customer/session context, and model ensembles.
- Add persistence and audit trails around the structured response models.

