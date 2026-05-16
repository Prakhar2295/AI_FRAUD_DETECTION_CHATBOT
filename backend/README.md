# Banking Fraud Detection Voice AI

This backend is being built phase by phase. Keep phase-specific implementation notes in separate README files so earlier architecture decisions are preserved instead of overwritten.

## Phase Documents

- [Phase 1: Offline Voice Intelligence Pipeline](README_Phase_1.md)
- [Phase 2: Live Voice + LangGraph Conversational Intelligence](README_Phase_2.md)
- [Phase 3: FastAPI WebSocket Realtime Streaming Infrastructure](README_Phase_3.md)

## Current Phase

The current implementation is Phase 3:

```text
Microphone Audio Stream
-> Audio Chunking
-> FastAPI WebSocket Streaming
-> Streaming STT Processing
-> LangGraph workflow execution
-> Streaming Fraud Intelligence Output
```

## Quick Start

From the project root:

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
python backend/app/main.py
```

Make sure Ollama is running with the local model:

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

The realtime WebSocket endpoint is:

```text
ws://127.0.0.1:8000/ws/voice/{session_id}
```

## Documentation Rule

For future phases, create a new file such as `README_Phase_4.md` and add it to the phase list above. Keep this `README.md` as the stable index and quick-start page.

