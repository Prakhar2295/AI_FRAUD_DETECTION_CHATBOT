"""Realtime streaming orchestration pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

from app.config.settings import Settings
from app.graph.fraud_workflow import build_fraud_workflow
from app.graph.state import create_initial_state
from app.models.response_models import RiskAnalysisResponse, WorkflowOutputResponse
from app.models.websocket_models import (
    StreamingTranscriptionUpdate,
    WebSocketOutboundMessage,
)
from app.realtime.audio_stream_manager import AudioStreamManager
from app.realtime.chunk_processor import ChunkProcessor, TranscriptionWindow
from app.realtime.vad_service import VADService
from app.realtime.websocket_manager import WebSocketManager
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.services.stt_service import STTService
from app.utils.logger import get_logger


class StreamingPipeline:
    """Coordinate audio queues, pseudo-streaming STT, LangGraph, and output."""

    def __init__(
        self,
        session_id: str,
        settings: Settings,
        websocket_manager: WebSocketManager,
        stt_service: STTService,
        llm_service: OllamaLLMService,
        memory_service: MemoryService,
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.websocket_manager = websocket_manager
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.audio_stream = AudioStreamManager(settings, session_id)
        self.chunk_processor = ChunkProcessor(settings, VADService(settings))
        self.logger = get_logger(f"{self.__class__.__name__}.{session_id}")
        self._running = False
        self._accumulated_transcript = ""
        self._workflow_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        """Consume audio queue items until stopped."""
        self._running = True
        self.logger.info("Streaming pipeline started")
        try:
            while self._running:
                item = await self.audio_stream.next_item()

                if item.kind == "stop":
                    await self._process_flush()
                    break
                if item.kind == "flush":
                    await self._process_flush()
                    continue

                if item.payload is None or item.metadata is None:
                    await self._send_error("Malformed audio queue item")
                    continue

                try:
                    window = self.chunk_processor.process_chunk(
                        item.payload,
                        item.metadata,
                    )
                except ValueError as exc:
                    await self._send_error(str(exc))
                    continue

                if window is not None:
                    await self._process_window(window)
        finally:
            self._running = False
            await self.websocket_manager.send_message(
                self.session_id,
                WebSocketOutboundMessage(
                    type="session_closed",
                    session_id=self.session_id,
                    payload={"reason": "pipeline_stopped"},
                ),
            )
            self.logger.info("Streaming pipeline stopped")

    async def stop(self) -> None:
        """Request pipeline shutdown."""
        self._running = False
        await self.audio_stream.enqueue_stop()

    async def enqueue_audio(self, payload: bytes, sequence: int) -> None:
        """Queue a binary PCM16 audio chunk for processing."""
        from app.models.websocket_models import AudioChunkMetadata

        metadata = AudioChunkMetadata(
            sequence=sequence,
            sample_rate=self.settings.audio_sample_rate,
            channels=self.settings.audio_channels,
            encoding="pcm_s16le",
        )
        try:
            await self.audio_stream.enqueue_audio(payload, metadata)
            await self.websocket_manager.send_message(
                self.session_id,
                WebSocketOutboundMessage(
                    type="ack",
                    session_id=self.session_id,
                    payload={"sequence": sequence, "queue_size": self.audio_stream.queue.qsize()},
                ),
            )
        except asyncio.QueueFull:
            await self._send_error("Audio queue overflow")

    async def flush(self) -> None:
        """Flush buffered audio into a transcription window."""
        await self.audio_stream.enqueue_flush()

    async def _process_flush(self) -> None:
        window = self.chunk_processor.flush()
        if window is not None:
            await self._process_window(window)

    async def _process_window(self, window: TranscriptionWindow) -> None:
        self.logger.info(
            "Processing transcription window: start=%s end=%s bytes=%s",
            window.start_sequence,
            window.end_sequence,
            len(window.payload),
        )
        try:
            transcription = await self.stt_service.transcribe_pcm_window_async(
                pcm_audio=window.payload,
                sample_rate=window.sample_rate,
                channels=window.channels,
                temp_dir=self.settings.temp_audio_dir,
            )
        except Exception as exc:
            await self._send_error(f"Transcription failed: {exc}")
            return

        partial = transcription.text.strip()
        if not partial:
            self.logger.info("Skipping empty transcription window")
            return

        self._accumulated_transcript = " ".join(
            part for part in [self._accumulated_transcript, partial] if part
        ).strip()
        await self._send_transcription_update(partial, window.end_sequence)
        await self._execute_workflow(partial, window.end_sequence)

    async def _execute_workflow(self, partial: str, sequence: int) -> None:
        workflow = build_fraud_workflow(
            llm_service=self.llm_service,
            memory_service=self.memory_service,
            settings=self.settings,
        )
        initial_state = create_initial_state(
            session_id=self.session_id,
            transcript=self._accumulated_transcript,
            partial_transcript=partial,
            stream_sequence=sequence,
            workflow_history=self._workflow_history,
        )

        try:
            final_state = await asyncio.to_thread(workflow.invoke, initial_state)
            output = self._workflow_output(final_state)
            self._workflow_history.append(self._model_to_dict(output))
            await self.websocket_manager.send_message(
                self.session_id,
                WebSocketOutboundMessage(
                    type="fraud_intelligence",
                    session_id=self.session_id,
                    payload=self._model_to_dict(output),
                ),
            )
            await self.websocket_manager.send_message(
                self.session_id,
                WebSocketOutboundMessage(
                    type="session_state",
                    session_id=self.session_id,
                    payload={
                        "sequence": sequence,
                        "workflow_history_count": len(self._workflow_history),
                    },
                ),
            )
        except Exception as exc:
            await self._send_error(f"Workflow execution failed: {exc}")

    async def _send_transcription_update(self, partial: str, sequence: int) -> None:
        update = StreamingTranscriptionUpdate(
            session_id=self.session_id,
            transcript=partial,
            accumulated_transcript=self._accumulated_transcript,
            sequence=sequence,
            is_final=False,
        )
        await self.websocket_manager.send_message(
            self.session_id,
            WebSocketOutboundMessage(
                type="transcription_update",
                session_id=self.session_id,
                payload=self._model_to_dict(update),
            ),
        )

    async def _send_error(self, message: str) -> None:
        self.logger.error(message)
        await self.websocket_manager.send_message(
            self.session_id,
            WebSocketOutboundMessage(
                type="error",
                session_id=self.session_id,
                payload={"message": message},
            ),
        )

    def _workflow_output(self, state: dict[str, Any]) -> WorkflowOutputResponse:
        risk = RiskAnalysisResponse(**state["risk"])
        history = state.get("conversation_history", [])
        return WorkflowOutputResponse(
            session_id=state["session_id"],
            transcript=state["transcript"],
            partial_transcript=state.get("partial_transcript"),
            stream_sequence=state.get("stream_sequence"),
            intent_classification=state.get("intent"),
            suspicious_indicators=state.get("suspicious_indicators", []),
            fraud_risk_score=risk.fraud_risk_score,
            risk_level=risk.risk_level,
            reasoning_summary=risk.reasoning_summary,
            workflow_execution_trace=state.get("workflow_trace", []),
            node_execution_timestamps=state.get("node_timestamps", {}),
            conversation_turn_count=len(history),
            errors=state.get("errors", []),
        )

    @staticmethod
    def _model_to_dict(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

