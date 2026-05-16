# Banking Fraud Detection Voice AI - Phase 3

Phase 3 converts the local conversational workflow into realtime streaming infrastructure using FastAPI WebSockets and chunk-window audio processing.

```text
Microphone Audio Stream
-> Audio Chunking
-> FastAPI WebSocket Streaming
-> Streaming STT Processing
-> LangGraph Workflow Execution
-> Streaming Fraud Intelligence Output
```

This phase intentionally does not include frontend UI, browser audio capture, TTS, deepfake detection, replay detection, Redis, vector databases, Docker, cloud deployment, or behavioral biometrics.

## Architecture

```text
backend/
├── app/
│   ├── api/
│   │   └── websocket.py
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
│   ├── realtime/
│   │   ├── audio_stream_manager.py
│   │   ├── websocket_manager.py
│   │   ├── chunk_processor.py
│   │   ├── streaming_pipeline.py
│   │   └── vad_service.py
│   ├── services/
│   │   ├── microphone_service.py
│   │   ├── stt_service.py
│   │   ├── llm_service.py
│   │   ├── memory_service.py
│   │   └── fraud_analysis_service.py
│   ├── models/
│   │   ├── websocket_models.py
│   │   ├── conversation_models.py
│   │   └── response_models.py
│   ├── utils/
│   │   └── logger.py
│   └── main.py
├── data/
├── requirements.txt
├── README.md
├── README_Phase_1.md
├── README_Phase_2.md
└── README_Phase_3.md
```

## Setup

From the project root:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Make sure Ollama is running:

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

## Run WebSocket Server

```bash
source venv/bin/activate
python backend/app/main.py
```

Default endpoint:

```text
ws://127.0.0.1:8000/ws/voice/{session_id}
```

Health check:

```text
http://127.0.0.1:8000/health
```

To run the Phase 2 microphone CLI path instead:

```bash
RUN_MODE=cli python backend/app/main.py
```

## WebSocket Protocol

Server sends a `session_started` message after connection.

Binary messages are interpreted as raw PCM16 little-endian mono audio chunks:

```text
encoding: pcm_s16le
sample_rate: 16000
channels: 1
```

JSON control messages:

```json
{"type": "ping"}
{"type": "flush"}
{"type": "stop"}
```

Server output message types:

- `session_started`
- `ack`
- `transcription_update`
- `fraud_intelligence`
- `session_state`
- `error`
- `session_closed`

## Configuration

Optional environment variables:

```bash
export WEBSOCKET_HOST=127.0.0.1
export WEBSOCKET_PORT=8000
export AUDIO_SAMPLE_RATE=16000
export AUDIO_CHANNELS=1
export AUDIO_CHUNK_SIZE_BYTES=3200
export AUDIO_WINDOW_SECONDS=3
export AUDIO_QUEUE_MAX_SIZE=32
export VAD_ENERGY_THRESHOLD=0.01
export VAD_MIN_SPEECH_CHUNKS=1
export WHISPER_MODEL_SIZE=base
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
export OLLAMA_MODEL=qwen2.5:1.5b
export OLLAMA_ENDPOINT=http://localhost:11434/api/generate
export OLLAMA_TIMEOUT_SECONDS=120
```

## Testing

Compile check:

```bash
venv/bin/python -m compileall backend/app
```

Dependency check:

```bash
venv/bin/python -m pip check
```

In-process WebSocket lifecycle check:

```bash
venv/bin/python -c 'import sys; sys.path.insert(0, "backend"); from fastapi.testclient import TestClient; from app.main import create_app; client=TestClient(create_app()); ctx=client.websocket_connect("/ws/voice/test-session"); ws=ctx.__enter__(); print(ws.receive_json()); ws.send_json({"type":"ping"}); print(ws.receive_json()); ws.send_json({"type":"stop"}); print(ws.receive_json()); ctx.__exit__(None, None, None)'
```

## Design Notes

### WebSocket Architecture

`api/websocket.py` owns the FastAPI endpoint and connection lifecycle. It accepts binary audio chunks and simple JSON control messages. Heavy work is delegated to `StreamingPipeline`.

### Queue Architecture

`AudioStreamManager` owns a session-scoped `asyncio.Queue`. The WebSocket handler enqueues quickly and returns to receiving messages. Queue overflow returns an `error` message instead of blocking indefinitely.

### Streaming Orchestration

`StreamingPipeline` consumes queued chunks, passes them through `ChunkProcessor`, triggers transcription windows, executes LangGraph, and emits structured streaming outputs.

### STT Strategy

This phase uses pseudo-streaming STT. Audio chunks are aggregated into short PCM windows, materialized as temporary WAV files, and transcribed with faster-whisper. This avoids pretending we have token-level realtime STT while keeping the architecture ready for lower-latency upgrades.

### LangGraph Realtime Adaptation

The graph remains deterministic:

```text
intent -> fraud -> risk -> memory -> END
```

Workflow state now supports partial transcripts, accumulated transcripts, stream sequence numbers, and workflow history. Nodes still communicate only through explicit serializable state.

### Future Paths

- TTS integration: add an outbound audio queue after `fraud_intelligence` without changing inbound audio handling.
- Deepfake detection: add a future audio-authentication service before STT or as a parallel realtime node.
- Scaling: move per-session queues and memory to external infrastructure later; keep the current service contracts stable.
- Bottlenecks: faster-whisper transcription windows and Ollama calls are the most likely latency sources.
- Observability: add per-window latency metrics, queue depth metrics, trace IDs, and structured JSON logs in a later phase.
