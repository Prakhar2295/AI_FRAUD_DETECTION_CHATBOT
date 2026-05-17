"""Session-aware fraud memory storage."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.models.conversation_models import ConversationInteraction, ConversationSession
from app.utils.logger import get_logger


class SessionMemory:
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self.logger = get_logger(self.__class__.__name__)

    def append_interaction(
        self,
        session_id: str,
        interaction: ConversationInteraction,
    ) -> ConversationSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = ConversationSession(session_id=session_id)

        session.interactions.append(interaction)
        self._sessions[session_id] = session
        self.logger.info(
            "SessionMemory append_interaction session_id=%s interactions=%s",
            session_id,
            len(session.interactions),
        )
        return deepcopy(session)

    def get_session(self, session_id: str) -> ConversationSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = ConversationSession(session_id=session_id)
            self._sessions[session_id] = session
        return deepcopy(session)

    def get_history(self, session_id: str) -> list[ConversationInteraction]:
        return self.get_session(session_id).interactions

    def get_recent_transcripts(self, session_id: str, limit: int = 8) -> list[str]:
        session = self.get_session(session_id)
        return [interaction.transcript for interaction in session.interactions[-limit:]]

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "interaction_count": len(session.interactions),
            "last_transcript": session.interactions[-1].transcript if session.interactions else None,
        }
