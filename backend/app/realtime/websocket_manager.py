"""WebSocket connection lifecycle manager."""

from __future__ import annotations

from fastapi import WebSocket

from app.models.websocket_models import WebSocketOutboundMessage
from app.utils.logger import get_logger


class WebSocketManager:
    """Track active realtime client connections."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self.logger = get_logger(self.__class__.__name__)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Accept and register a websocket connection."""
        await websocket.accept()
        self._connections[session_id] = websocket
        self.logger.info("WebSocket connected: session_id=%s", session_id)

    def disconnect(self, session_id: str) -> None:
        """Remove a websocket connection from active tracking."""
        self._connections.pop(session_id, None)
        self.logger.info("WebSocket disconnected: session_id=%s", session_id)

    async def send_message(
        self,
        session_id: str,
        message: WebSocketOutboundMessage,
    ) -> None:
        """Send a typed JSON message to one client."""
        websocket = self._connections.get(session_id)
        if websocket is None:
            self.logger.warning("Cannot send; session is not connected: %s", session_id)
            return

        if hasattr(message, "model_dump"):
            payload = message.model_dump()
        else:
            payload = message.dict()
        try:
            await websocket.send_json(payload)
        except RuntimeError as exc:
            self.logger.warning(
                "Failed to send websocket message: session_id=%s error=%s",
                session_id,
                exc,
            )
