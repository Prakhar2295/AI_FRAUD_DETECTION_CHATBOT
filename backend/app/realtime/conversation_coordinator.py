"""Central orchestration for real-time conversation, workflow, and TTS playback."""

from __future__ import annotations

import asyncio
from typing import Any

from app.audio.audio_output_service import AudioOutputService
from app.audio.interruption_manager import InterruptionManager
from app.audio.turn_manager import TurnManager
from app.graph.fraud_workflow import build_fraud_workflow
from app.graph.state import create_initial_state, utc_now_iso
from app.models.response_models import AIResponse, RiskAnalysisResponse, WorkflowOutputResponse
from app.models.websocket_models import WebSocketOutboundMessage
from app.realtime.websocket_manager import WebSocketManager
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.tts.streaming_tts import StreamingTTS
from app.tts.tts_service import TTSService
from app.tts.voice_config import VoiceConfig
from app.utils.logger import get_logger


class ConversationCoordinator:
    """Coordinate the streaming pipeline, LangGraph workflow, TTS, and playback."""

    def __init__(
        self,
        session_id: str,
        settings: Any,
        websocket_manager: WebSocketManager,
        llm_service: OllamaLLMService,
        memory_service: MemoryService,
        tts_service: TTSService,
        audio_output_service: AudioOutputService,
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.websocket_manager = websocket_manager
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.tts_service = tts_service
        self.audio_output_service = audio_output_service
        self.streaming_tts = StreamingTTS(tts_service, chunk_size=settings.audio_chunk_size_bytes)
        self.turn_manager = TurnManager()
        self.interruption_manager = InterruptionManager(audio_output_service.playback_manager)
        self.logger = get_logger(self.__class__.__name__)
        self._lock = asyncio.Lock()
        self._workflow_history: list[dict[str, Any]] = []
        self._conversation_metadata: dict[str, Any] = {
            "session_started_at": utc_now_iso(),
            "voice_response_enabled": True,
            "tts_voice": settings.tts_voice_name,
        }

    async def handle_transcription_window(
        self,
        transcript: str,
        sequence: int,
        accumulated_transcript: str,
        fraud_metadata: dict | None = None,
        behavioral_metadata: dict | None = None,
    ) -> None:
        async with self._lock:
            self.turn_manager.start_user_turn()
            self.logger.info(
                "Handling transcription window sequence=%s transcript=%s",
                sequence,
                transcript,
            )

            workflow = build_fraud_workflow(
                llm_service=self.llm_service,
                memory_service=self.memory_service,
                settings=self.settings,
            )
            initial_state = create_initial_state(
                session_id=self.session_id,
                transcript=accumulated_transcript,
                partial_transcript=transcript,
                stream_sequence=sequence,
                workflow_history=self._workflow_history,
            )

            if fraud_metadata is not None:
                initial_state["fraud_audio"] = fraud_metadata
            if behavioral_metadata is not None:
                initial_state["behavioral"] = behavioral_metadata
                initial_state["behavioral_metadata"] = behavioral_metadata

            try:
                final_state = await asyncio.to_thread(workflow.invoke, initial_state)
            except Exception as exc:
                self.logger.error("Workflow invocation failed: %s", exc)
                await self._publish_tts_status("error", str(exc))
                await self.websocket_manager.send_message(
                    self.session_id,
                    WebSocketOutboundMessage(
                        type="error",
                        session_id=self.session_id,
                        payload={
                            "message": "Workflow execution failed, continuing conversation with fallback state.",
                            "detail": str(exc),
                        },
                    ),
                )
                self.turn_manager.complete_user_turn()
                return

            final_state["active_speaker"] = "assistant"
            final_state["playback_state"] = "queued"
            final_state["conversation_turns"] = self.turn_manager.turn_count + 1
            final_state["interruption_state"] = self.interruption_manager.snapshot()
            final_state["session_metadata"] = self._conversation_metadata

            output = self._workflow_output(final_state)
            self._workflow_history.append(self._model_to_dict(output))

            await self._publish_fraud_intelligence(output, sequence)
            await self._publish_session_state(final_state, sequence)
            await self._publish_ai_response(output, sequence)

            self.turn_manager.complete_user_turn()
            self.turn_manager.start_ai_turn()
            await self._enqueue_tts_playback(output)
            self.turn_manager.complete_ai_turn()

    async def _publish_fraud_intelligence(
        self,
        output: WorkflowOutputResponse,
        sequence: int,
    ) -> None:
        payload = self._model_to_dict(output)
        await self.websocket_manager.send_message(
            self.session_id,
            WebSocketOutboundMessage(
                type="fraud_intelligence",
                session_id=self.session_id,
                payload={"sequence": sequence, "workflow_output": payload},
            ),
        )

    async def _publish_session_state(self, state: dict[str, Any], sequence: int) -> None:
        await self.websocket_manager.send_message(
            self.session_id,
            WebSocketOutboundMessage(
                type="session_state",
                session_id=self.session_id,
                payload={
                    "sequence": sequence,
                    "conversation_turn_count": state.get("conversation_turns", 0),
                    "active_speaker": state.get("active_speaker"),
                    "playback_state": state.get("playback_state"),
                    "interruption_state": state.get("interruption_state"),
                    "workflow_history_count": len(self._workflow_history),
                },
            ),
        )

    async def _publish_ai_response(
        self,
        output: WorkflowOutputResponse,
        sequence: int,
    ) -> None:
        if output.ai_response is None:
            return

        await self.websocket_manager.send_message(
            self.session_id,
            WebSocketOutboundMessage(
                type="ai_response",
                session_id=self.session_id,
                payload={
                    "sequence": sequence,
                    "response": self._model_to_dict(output.ai_response),
                    "turn_count": self.turn_manager.turn_count + 1,
                },
            ),
        )

    async def _enqueue_tts_playback(self, output: WorkflowOutputResponse) -> None:
        if output.ai_response is None or not output.ai_response.response_text:
            self.logger.info("No AI response to play back")
            return

        voice_config = VoiceConfig(
            voice_name=self.settings.tts_voice_name,
            style=output.ai_response.voice_style or self.settings.tts_voice_style,
            language=self.settings.tts_language,
            sample_rate=self.settings.audio_sample_rate,
        )

        buffer = []
        try:
            async for chunk in self.streaming_tts.stream_text(
                output.ai_response.response_text,
                voice_config,
            ):
                buffer.append(chunk.pcm_bytes)

            audio_bytes = b"".join(buffer)
            await self.audio_output_service.enqueue_audio(
                pcm_bytes=audio_bytes,
                sample_rate=voice_config.sample_rate,
                channels=voice_config.channels,
            )
            await self._publish_tts_status("queued")
        except Exception as exc:
            self.logger.error("TTS playback enqueue failed: %s", exc)
            await self._publish_tts_status("error", str(exc))

    async def _publish_tts_status(self, status: str, reason: str | None = None) -> None:
        await self.websocket_manager.send_message(
            self.session_id,
            WebSocketOutboundMessage(
                type="tts_status",
                session_id=self.session_id,
                payload={
                    "state": status,
                    "queue_size": self.audio_output_service.playback_manager.queue.qsize(),
                    "reason": reason,
                },
            ),
        )

    def _workflow_output(self, state: dict[str, Any]) -> WorkflowOutputResponse:
        risk = RiskAnalysisResponse(**state["risk"])
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
            conversation_turn_count=state.get("conversation_turns", 0),
            retrieved_fraud_patterns=state.get("retrieved_fraud_patterns", []),
            semantic_retrieval_metadata=state.get("semantic_retrieval_metadata", {}),
            historical_context=state.get("historical_fraud_context"),
            adaptive_risk_enrichment=state.get("adaptive_risk_enrichment", {}),
            fraud_knowledge_context=state.get("fraud_knowledge_context"),
            behavioral=state.get("behavioral"),
            ai_response=state.get("ai_response"),
            errors=state.get("errors", []),
        )

    @staticmethod
    def _model_to_dict(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()
