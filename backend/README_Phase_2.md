# Banking Fraud Detection Voice AI - Phase 2

Phase 2 builds a modular live voice conversational intelligence pipeline:

```text
Microphone Input
-> Live Audio Capture
-> faster-whisper speech-to-text
-> LangGraph workflow execution
-> Intent analysis
-> Fraud analysis
-> Risk scoring
-> Conversation memory tracking
-> Structured fraud intelligence output
```

This phase intentionally does not include WebSocket streaming, frontend, TTS, deepfake detection, replay detection, Docker, or cloud deployment.

## Architecture

```text
backend/
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── graph/
│   │   ├── fraud_workflow.py
│   │   ├── state.py
│   │   └── nodes/
│   │       ├── intent_node.py
│   │       ├── fraud_node.py
│   │       ├── risk_node.py
│   │       └── memory_node.py
│   ├── services/
│   │   ├── microphone_service.py
│   │   ├── stt_service.py
│   │   ├── llm_service.py
│   │   ├── memory_service.py
│   │   └── fraud_analysis_service.py
│   ├── models/
│   │   ├── conversation_models.py
│   │   └── response_models.py
│   ├── utils/
│   │   └── logger.py
│   └── main.py
├── data/
│   ├── fraud_detection_sample.wav
│   └── tmp/
├── requirements.txt
├── README.md
├── README_Phase_1.md
└── README_Phase_2.md
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

## Run

Activate the virtual environment first, then run:

```bash
python backend/app/main.py
```

The app records microphone audio, saves a temporary WAV file under `backend/data/tmp`, transcribes it, runs the LangGraph workflow, and prints structured fraud intelligence JSON.

On macOS, allow microphone permission for the terminal or VS Code when prompted.

## Configuration

Optional environment variables:

```bash
export WHISPER_MODEL_SIZE=base
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
export OLLAMA_MODEL=qwen2.5:1.5b
export OLLAMA_ENDPOINT=http://localhost:11434/api/generate
export OLLAMA_TIMEOUT_SECONDS=120
export RECORDING_DURATION_SECONDS=5
export AUDIO_SAMPLE_RATE=16000
export AUDIO_CHANNELS=1
export TEMP_AUDIO_DIR=backend/data/tmp
export LOW_RISK_MAX_SCORE=30
export MEDIUM_RISK_MAX_SCORE=70
export SESSION_ID=local-demo-session
```

## Service Responsibilities

- `MicrophoneService`: captures live microphone input and writes temporary WAV files.
- `STTService`: owns faster-whisper model loading and WAV transcription.
- `OllamaLLMService`: owns local Ollama API calls, timeout handling, and JSON parsing.
- `MemoryService`: stores session-scoped conversation history behind a Redis-ready abstraction.
- `FraudAnalysisService`: preserves the Phase 1 direct fraud analysis path.
- `response_models.py`: defines structured transcript, intent, fraud, risk, and workflow output contracts.
- `conversation_models.py`: defines session memory and workflow state snapshots.
- `logger.py`: provides timestamped service-level logging.

## LangGraph Workflow

The workflow is deterministic:

```text
intent -> fraud -> risk -> memory -> END
```

- `Intent Node`: classifies customer intent and transaction/request type.
- `Fraud Node`: detects suspicious indicators, urgency manipulation, emotional pressure, and suspicious intent.
- `Risk Node`: deterministically scores fraud risk from node outputs and configured thresholds.
- `Memory Node`: appends the processed interaction to session memory and records the final workflow state.

Each node receives workflow state and returns explicit state updates. Nodes communicate only through the shared typed state.

## Output

The final JSON includes:

- transcript
- intent classification
- suspicious indicators
- fraud risk score
- risk level
- reasoning summary
- workflow execution trace
- node execution timestamps
- conversation turn count
- workflow errors, if any

## Future Scalability Path

This structure is ready for the next phases:

- Realtime streaming: feed partial transcripts into the same LangGraph state shape over WebSockets.
- Multi-agent expansion: split nodes into specialized agents for intent, fraud tactics, policy, and customer safety.
- Redis integration: replace `MemoryService` internals with Redis while keeping the same service interface.
- FastAPI integration: expose microphone/file workflows as API routes without changing core services.
- Fraud expansion: add rules, retrieval, customer/session context, and audit persistence around the current models.
