"""FastAPI WebSocket routes for realtime voice streaming."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config.settings import Settings
from app.models.websocket_models import InboundControlMessage, WebSocketOutboundMessage
from app.realtime.streaming_pipeline import StreamingPipeline
from app.realtime.websocket_manager import WebSocketManager
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService
from app.services.stt_service import STTService
from app.utils.logger import get_logger


def create_websocket_router(settings: Settings) -> APIRouter:
    """Create websocket routes with app-scoped service instances."""
    router = APIRouter()
    logger = get_logger("WebSocketAPI")
    websocket_manager = WebSocketManager()
    stt_service = STTService(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    llm_service = OllamaLLMService(
        model_name=settings.ollama_model,
        endpoint=settings.ollama_endpoint,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    memory_service = MemoryService()

    @router.websocket("/ws/voice/{session_id}")
    async def voice_stream(websocket: WebSocket, session_id: str) -> None:
        await websocket_manager.connect(session_id, websocket)
        pipeline = StreamingPipeline(
            session_id=session_id,
            settings=settings,
            websocket_manager=websocket_manager,
            stt_service=stt_service,
            llm_service=llm_service,
            memory_service=memory_service,
        )
        pipeline_task = asyncio.create_task(pipeline.run())
        sequence = 0

        await websocket_manager.send_message(
            session_id,
            WebSocketOutboundMessage(
                type="session_started",
                session_id=session_id,
                payload={
                    "sample_rate": settings.audio_sample_rate,
                    "channels": settings.audio_channels,
                    "encoding": "pcm_s16le",
                    "audio_window_seconds": settings.audio_window_seconds,
                },
            ),
        )

        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                binary_payload = message.get("bytes")
                text_payload = message.get("text")

                if binary_payload is not None:
                    sequence += 1
                    await pipeline.enqueue_audio(binary_payload, sequence)
                    continue

                if text_payload is not None:
                    should_continue = await _handle_control_message(
                        text_payload,
                        pipeline,
                        websocket_manager,
                        session_id,
                    )
                    if not should_continue:
                        break
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: session_id=%s", session_id)
        finally:
            await pipeline.stop()
            await pipeline_task
            websocket_manager.disconnect(session_id)

    return router


async def _handle_control_message(
    text_payload: str,
    pipeline: StreamingPipeline,
    websocket_manager: WebSocketManager,
    session_id: str,
) -> bool:
    """Handle JSON control messages from the websocket client."""
    try:
        message = InboundControlMessage.model_validate_json(text_payload)
    except ValueError as exc:
        await websocket_manager.send_message(
            session_id,
            WebSocketOutboundMessage(
                type="error",
                session_id=session_id,
                payload={"message": f"Malformed control message: {exc}"},
            ),
        )
        return True

    if message.type == "ping":
        await websocket_manager.send_message(
            session_id,
            WebSocketOutboundMessage(
                type="ack",
                session_id=session_id,
                payload={"message": "pong"},
            ),
        )
        return True

    if message.type == "flush":
        await pipeline.flush()
        return True

    if message.type == "stop":
        return False

    await websocket_manager.send_message(
        session_id,
        WebSocketOutboundMessage(
            type="error",
            session_id=session_id,
            payload={"message": f"Unsupported control message: {message.type}"},
        ),
    )
    return True

