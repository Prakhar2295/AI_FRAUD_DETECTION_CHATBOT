"""Long-term conversation history helpers."""

from __future__ import annotations

from app.memory.session_memory import SessionMemory
from app.models.conversation_models import ConversationInteraction


class ConversationHistoryManager:
    """Read-oriented helper for session conversation continuity."""

    def __init__(self, session_memory: SessionMemory) -> None:
        self.session_memory = session_memory

    def recent_interactions(
        self,
        session_id: str,
        limit: int = 5,
    ) -> list[ConversationInteraction]:
        return self.session_memory.get_history(session_id)[-limit:]

    def recent_transcript_context(self, session_id: str, limit: int = 5) -> str:
        interactions = self.recent_interactions(session_id, limit)
        return "\n".join(interaction.transcript for interaction in interactions)
