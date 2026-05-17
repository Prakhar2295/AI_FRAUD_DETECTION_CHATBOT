"""Persistent fraud memory service with retrieval and adaptive enrichment."""

from __future__ import annotations

from typing import Any

from app.memory.fraud_memory_manager import FraudMemoryManager
from app.models.conversation_models import ConversationInteraction, ConversationSession
from app.utils.logger import get_logger


class MemoryService:
    """Facade service that exposes session memory and adaptive fraud memory APIs."""

    def __init__(self) -> None:
        self.manager = FraudMemoryManager()
        self.logger = get_logger(self.__class__.__name__)

    async def start(self) -> None:
        await self.manager.start()

    async def stop(self) -> None:
        await self.manager.stop()

    async def persist_interaction(
        self,
        session_id: str,
        interaction: ConversationInteraction,
        embedding_text: str | None = None,
    ) -> ConversationSession:
        """Persist one interaction asynchronously and return a session snapshot."""
        await self.manager.persist_interaction(
            session_id=session_id,
            interaction=interaction,
            embedding_text=embedding_text,
        )
        return self.get_session(session_id)

    def append_interaction(
        self,
        session_id: str,
        interaction: ConversationInteraction,
    ) -> ConversationSession:
        """Persist one interaction synchronously to session memory."""
        embedding_text = self._build_embedding_text(interaction)
        self.manager.persist_interaction_sync(session_id, interaction, embedding_text)
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> ConversationSession:
        """Return a session snapshot, creating an empty session if needed."""
        return self.manager.get_session(session_id)

    def get_history(self, session_id: str) -> list[ConversationInteraction]:
        """Return prior interactions for a session."""
        return self.manager.get_history(session_id)

    def retrieve_similar_fraud(self, session_id: str, transcript: str) -> dict[str, Any]:
        return self.manager.retrieve_similar_fraud(session_id=session_id, transcript=transcript)

    def enrich_risk(
        self,
        session_id: str,
        base_risk: dict[str, Any],
        retrieval_context: dict[str, Any],
    ) -> dict[str, Any]:
        return self.manager.enrich_risk(
            session_id=session_id,
            base_risk=base_risk,
            retrieval_context=retrieval_context,
        )

    @staticmethod
    def _build_embedding_text(interaction: ConversationInteraction) -> str:
        pieces = [
            interaction.transcript,
            str(interaction.fraud),
            str(interaction.behavioral),
            str(interaction.fraud_audio),
            str(interaction.retrieved_fraud_patterns),
        ]
        return "\n".join(piece for piece in pieces if piece and piece != "None")
