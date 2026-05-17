# Banking Fraud Detection Voice AI - Phase 5 (Audio Fraud Intelligence)

Phase 5 introduces a modular, research-friendly Audio Fraud Intelligence layer focused on voice authenticity and audio attack detection. This phase is intentionally limited to audio fraud intelligence: replay, spoofing, and deepfake detection, feature extraction, and deterministic authenticity scoring.

Goals
- Add a modular fraud audio analysis pipeline that runs alongside the realtime conversational pipeline.
- Keep fraud audio intelligence independent from LangGraph business logic and WebSocket transport.
- Provide async-safe, queue-based orchestration for streaming analysis.
- Expose structured fraud metadata for enrichment of the LangGraph workflow.

High-level design

Realtime audio stream → Fraud Audio Pipeline (async queue)
  ├─ Feature extraction (MFCC, spectral, temporal)
  ├─ Replay detection (lightweight DSP + heuristics)
  ├─ Spoof detection (model abstraction, pluggable backends)
  ├─ Deepfake analysis (spectral & consistency checks)
  └─ Authenticity engine (aggregator, deterministic scoring)

Repository additions

The following files were added under `backend/app/fraud_audio/`:

- `audio_feature_extractor.py` — reusable feature extraction helpers (MFCC, spectral, waveform)
- `replay_detector.py` — replay attack detection interface and lightweight heuristic implementation
- `spoof_detector.py` — spoof detection abstraction and placeholder implementation
- `deepfake_detector.py` — deepfake analysis abstraction and placeholder implementation
- `artifact_analyzer.py` — detection of audio artifacts and corruption
- `voice_risk_aggregator.py` — deterministic scoring and aggregation of signals
- `authenticity_engine.py` — orchestration glue that invokes detectors and aggregates results
- `fraud_audio_pipeline.py` — async-safe pipeline coordinator that accepts streaming audio and schedules analysis

Async and runtime notes

- The fraud audio pipeline uses `asyncio.Queue` to avoid blocking WebSocket handlers.
- Detector implementations are intentionally lightweight and research-friendly; heavy model inference should be isolated to separate processes or GPU workers in future work.
- All detectors return structured dictionaries with confidence scores and metadata; the `AuthenticityEngine` consumes these and returns a deterministic `authenticity_score` and detailed `fraud_audio_metadata`.

Integration points

- The `FraudAudioPipeline` exposes an async `enqueue_audio(session_id, pcm16_bytes, timestamp)` method for the realtime pipeline to call. It returns a `asyncio.Task` that resolves to detector outputs.
- `AuthenticityNode` (to be added to LangGraph) will ingest `fraud_audio_metadata` and `authenticity_score` to enrich risk calculations.

Testing and validation

- Unit tests should validate feature extraction and replay/spoof heuristics using short WAV fixtures.
- Integration tests should run the `FraudAudioPipeline` with a sample WAV and assert output schema and non-empty confidence scores.

Setup

Install dependencies (see top-level `requirements.txt` which now includes `librosa`, `resampy`, and `python_speech_features`). On Apple Silicon, prefer installing in a Python 3.14 venv.

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Next steps

- Add an `AuthenticityNode` to the LangGraph workflow that consumes outputs from this pipeline.
- Replace placeholder detector implementations with research-grade models (AASIST, RawNet2, ASVspoof-compatible models) as experiments progress.
- Add GPU worker integration and model-serving interface for heavy inference.

Research & safety note

This phase deals with security-sensitive detection of synthetic and replayed audio. Keep models, thresholds, and scoring functions configurable and auditable. Do not make automated blocking decisions in production without human review and rigorous evaluation.
