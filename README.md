# Banking Fraud Detection Voice AI — Project Overview

This repository contains a multi-phase implementation of a Banking Fraud Detection Voice AI system. Each phase has its own detailed README under `backend/`.

Phase-specific docs:

- Phase 1: [backend/README_Phase_1.md](backend/README_Phase_1.md)
- Phase 2: [backend/README_Phase_2.md](backend/README_Phase_2.md)
- Phase 3: [backend/README_Phase_3.md](backend/README_Phase_3.md)
- Phase 4: [backend/README_Phase_4.md](backend/README_Phase_4.md)

Summary of phases:

- **Phase 1**: Offline batch processing pipeline (WAV → STT → Ollama LLM → fraud reasoning). See [backend/README_Phase_1.md](backend/README_Phase_1.md).
- **Phase 2**: Microphone and near-realtime streaming pipeline; local microphone capture and basic STT streaming.
- **Phase 3**: Graph-based orchestration using LangGraph; deterministic workflow for intent, fraud analysis, risk scoring, and memory.
- **Phase 4**: Realtime conversational voice responses, TTS integration, playback and turn/interruption management. See the Phase 4 section below and [backend/README_Phase_4.md](backend/README_Phase_4.md) for details.

# Banking Fraud Detection Voice AI — Phase 4

This workspace implements Phase 4 of a local Banking Fraud Detection Voice AI system.
Phase 4 extends the existing streaming fraud intelligence pipeline with realtime conversational voice responses, turn management, interruption-aware playback, and a modular TTS orchestration layer.

## What’s new in Phase 4

- Realtime conversational AI response generation
- Modular Piper-compatible TTS service architecture
- Queue-safe audio playback and stateful turn management
- Interruption-safe foundation for future barge-in support
- LangGraph workflow expanded with an AI response node
- Conversation coordinator for STT → workflow → TTS orchestration
- Session state updates for active speaker, playback status, and turn counts

## Architecture overview

- `backend/app/realtime/streaming_pipeline.py`
  - consumes microphone/WebSocket audio chunks
  - performs pseudo-streaming STT
  - delegates workflow and voice orchestration to `ConversationCoordinator`

- `backend/app/realtime/conversation_coordinator.py`
  - coordinates LangGraph execution, AI response generation, TTS synthesis, and audio playback
  - maintains deterministic turn state and session metadata

- `backend/app/graph/nodes/response_node.py`
  - adds AI response generation to the workflow
  - produces concise spoken responses with structured metadata

- `backend/app/tts/`
  - `tts_service.py` defines the TTS interface
  - `piper_service.py` provides a Piper-compatible implementation
  - `streaming_tts.py` provides chunked streaming wrappers
  - `voice_config.py` defines reusable voice settings

- `backend/app/audio/`
  - `audio_output_service.py` exposes async playback enqueueing
  - `playback_manager.py` drives queued playback safely
  - `interruption_manager.py` defines stop/interrupt behavior
  - `turn_manager.py` tracks speaker state and turn count
  - `response_buffer.py` accumulates streaming audio chunks

## Requirements

Python 3.14.5 compatible packages are listed in `requirements.txt`.
The new Phase 4 dependencies include:

- `piper-tts==0.0.6`
- `soundfile==0.12.1`
- `sounddevice` (already included)

> Note: Actual voice output depends on a Piper runtime or compatible TTS provider. If Piper is unavailable, the system falls back to placeholder audio while preserving orchestration logic.

## Setup

1. Create and activate a Python 3.14 environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Ensure Ollama is running locally and accessible at the URL configured via `OLLAMA_ENDPOINT`.
4. Optionally install Piper and make the `piper` CLI available on your PATH.

## Environment variables

The system uses environment variables with defaults in `backend/app/config/settings.py`.
Important values:

- `OLLAMA_ENDPOINT`
- `OLLAMA_MODEL`
- `AUDIO_SAMPLE_RATE`
- `AUDIO_CHANNELS`
- `AUDIO_WINDOW_SECONDS`
- `AUDIO_QUEUE_MAX_SIZE`
- `TTS_VOICE_NAME`
- `TTS_VOICE_STYLE`
- `TTS_LANGUAGE`
- `TTS_OUTPUT_DEVICE`

## Run the realtime server

From the repository root:

```bash
python backend/app/main.py
```

Or using Uvicorn directly:

```bash
uvicorn backend.app.main:create_app --host 127.0.0.1 --port 8000
```

## WebSocket test instructions

Connect to the realtime voice endpoint:

```text
ws://127.0.0.1:8000/ws/voice/<session_id>
```

Send binary PCM16 audio chunks and control messages:

- `flush` to force a transcription window
- `stop` to terminate the session
- `ping` to check liveness

Example control payload:

```json
{"type":"flush"}
```

Example responses from the server include:

- `session_started`
- `transcription_update`
- `fraud_intelligence`
- `ai_response`
- `tts_status`
- `session_state`
- `error`
- `session_closed`

## TTS setup notes

The Phase 4 TTS architecture is built to support Piper, but a working Piper runtime is required for real voice synthesis.

- Default voice: `alloy`
- Default language: `en`
- Default style: `neutral`

If no Piper CLI is available, the system will continue running with placeholder audio output so the conversation orchestration can still be developed and tested.

## What to test first

1. Start the server.
2. Open a WebSocket client to `/ws/voice/test-session`.
3. Send PCM16 audio frames and `flush` events.
4. Confirm the server emits transcription updates, fraud intelligence, AI response metadata, and TTS playback status.

## Phase 7: Persistent Fraud Memory and Adaptive Intelligence

Phase 7 introduces a modular persistent memory layer, vector-based fraud retrieval, adaptive risk enrichment, and cross-session fraud learning.

- Persistent session memory for long-term conversational context
- Vectorized fraud memory using ChromaDB semantics
- Fraud pattern retrieval and knowledge enrichment
- Adaptive risk scoring based on historical and retrieval signals
- Structured workflow metadata for retrieval and enrichment

## Phase 7 setup notes

1. Install the additional Phase 7 dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Ensure Ollama is running locally.
3. If `chromadb` is unavailable, the system falls back to an in-memory vector store for development.

## Memory and retrieval testing instructions

1. Start the server.
2. Send voice audio and observe workflow outputs.
3. Confirm the final response payload includes:
   - `retrieved_fraud_patterns`
   - `semantic_retrieval_metadata`
   - `historical_context`
   - `adaptive_risk_enrichment`
   - `fraud_knowledge_context`

4. Use repeated sessions to validate cross-session adaptive memory behavior.

## Future extension paths

- barge-in and duplex audio streaming
- deepfake and replay detection hooks
- behavioral fraud intelligence signals
- local voice cloning support via Piper
- richer session checkpointing and observability
