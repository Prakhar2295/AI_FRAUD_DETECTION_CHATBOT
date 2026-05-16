"""Session-scoped in-memory conversation storage."""

from __future__ import annotations

from copy import deepcopy

from app.models.conversation_models import ConversationInteraction, ConversationSession
from app.utils.logger import get_logger


class MemoryService:
    """Lightweight memory abstraction designed to be replaceable by Redis later."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self.logger = get_logger(self.__class__.__name__)

    def append_interaction(
        self,
        session_id: str,
        interaction: ConversationInteraction,
    ) -> ConversationSession:
        """Append one interaction to a session and return a snapshot."""
        session = self._sessions.get(session_id)
        if session is None:
            session = ConversationSession(session_id=session_id)

        session.interactions.append(interaction)
        self._sessions[session_id] = session
        self.logger.info(
            "Memory updated: session_id=%s interactions=%s",
            session_id,
            len(session.interactions),
        )
        return deepcopy(session)

    def get_session(self, session_id: str) -> ConversationSession:
        """Return a session snapshot, creating an empty session if needed."""
        session = self._sessions.get(session_id)
        if session is None:
            session = ConversationSession(session_id=session_id)
            self._sessions[session_id] = session
        return deepcopy(session)

    def get_history(self, session_id: str) -> list[ConversationInteraction]:
        """Return prior interactions for a session."""
        return self.get_session(session_id).interactions

