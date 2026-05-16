# Banking Fraud Detection Voice AI - Phase 4

Phase 4 transforms the offline pipeline into a realtime, full‑duplex conversational agent focused on realtime AI voice response and conversational orchestration.

This phase intentionally focuses ONLY on:
- realtime AI voice response (TTS)
- conversational orchestration and turn management
- interruption handling preparation
- integration of TTS playback into the realtime pipeline

## Architecture (high level)

```
Microphone / Client -> WebSocket -> StreamingPipeline -> ConversationCoordinator
    -> LangGraph (intent -> fraud -> risk -> response -> memory)
    -> Ollama LLM -> Streaming TTS (Piper or fallback)
    -> PlaybackManager / AudioOutputService -> speaker
```

Key components (under `backend/app`):

- `realtime/streaming_pipeline.py`: audio ingestion, STT windows, and handoff to the coordinator.
- `realtime/conversation_coordinator.py`: central orchestration (invoke workflow, publish websocket events, enqueue TTS playback).
- `graph/nodes/response_node.py`: LangGraph node to generate `ai_response` and metadata.
- `tts/`: TTS abstractions, `PiperTTSService` (CLI) and `PiperNotInstalledTTSService` fallback.
- `audio/playback_manager.py`, `audio/audio_output_service.py`: enqueue and play synthesized audio.
- `api/websocket.py`: WebSocket route `/ws/voice/{session_id}` for realtime audio and events.

## Dependencies and Piper note

The repository `requirements.txt` includes Python dependencies required to run the backend. Piper TTS has two integration paths:

- `piper-tts` Python package (optional) — pinned in `requirements.txt` to a PyPI-available version to enable programmatic Piper usage.
- Piper CLI (native binary) — for best-quality voices install the Piper CLI and voices separately. If Piper is not available the server uses a silent/fallback `PiperNotInstalledTTSService` so orchestration continues for testing.

To install Python deps:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want full TTS, install Piper CLI separately following Piper's docs and ensure the executable is on `PATH`.

## Running the realtime server

Start the backend server (from project root):

```bash
python3 backend/app/main.py
```

The server exposes the WebSocket endpoint at `ws://127.0.0.1:8000/ws/voice/{session_id}`.

## Quick websocket test (client sends resampled PCM16 frames)

The repo includes a simple test harness used during development. Example client behavior:

- Open the sample WAV at `backend/data/fraud_detection_sample.wav`.
- Resample to match pipeline windowing if needed.
- Connect to `ws://127.0.0.1:8000/ws/voice/test-session` and send raw PCM16 bytes.
- Send a JSON `{"type": "flush"}` to indicate end of utterance and `{"type": "stop"}` to terminate session.

A minimal Python client snippet (used during development) is available in the workspace; adapt it for tests.

## Observability and logs

Backend logs are emitted via `app.utils.logger.get_logger`. For server debugging, capture stdout/stderr to a log file and inspect it when the process exits unexpectedly.

```bash
python3 backend/app/main.py > /tmp/phase4_server.log 2>&1 &
tail -f /tmp/phase4_server.log
```

## Limitations & next steps

- Live TTS playback verification requires Piper CLI or another TTS runtime; otherwise the fallback produces silent audio so the orchestration and message flow can be validated.
- Ollama local server must be running for complete `ai_response` synthesis (install and run `ollama serve` before tests).
- Add structured telemetry and stricter health checks for external dependencies (Piper, Ollama).

## Where to look in code

- `backend/app/realtime/conversation_coordinator.py` — orchestration logic and event publishing
- `backend/app/graph/fraud_workflow.py` — LangGraph workflow including `response` node
- `backend/app/tts/` — TTS services and streaming helpers
- `backend/app/audio/` — playback and interruption management
- `backend/app/api/websocket.py` — WebSocket connection lifecycle and message schema

If you'd like, I can add a small runnable test script under `backend/tests/` that automates server startup, installs missing Python deps, and runs the websocket client used during development. Want me to add that next?
