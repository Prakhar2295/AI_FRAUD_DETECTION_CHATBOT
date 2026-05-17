"""High-level fraud memory manager for session and vector memory."""
from __future__ import annotations

from typing import Any

from app.memory.memory_coordinator import MemoryCoordinator
from app.models.conversation_models import ConversationInteraction
from app.utils.logger import get_logger


class FraudMemoryManager:
    def __init__(self) -> None:
        self.coordinator = MemoryCoordinator()
        self.logger = get_logger(self.__class__.__name__)

    async def start(self) -> None:
        await self.coordinator.start()

    async def stop(self) -> None:
        await self.coordinator.stop()

    async def persist_interaction(
        self,
        session_id: str,
        interaction: ConversationInteraction,
        embedding_text: str | None = None,
    ) -> None:
        await self.coordinator.persist_interaction(
            session_id=session_id,
            interaction=interaction,
            embedding_text=embedding_text,
        )

    def persist_interaction_sync(
        self,
        session_id: str,
        interaction: ConversationInteraction,
        embedding_text: str | None = None,
    ) -> None:
        self.coordinator.persist_interaction_sync(
            session_id=session_id,
            interaction=interaction,
            embedding_text=embedding_text,
        )

    def get_session(self, session_id: str) -> Any:
        return self.coordinator.session_memory.get_session(session_id)

    def get_history(self, session_id: str) -> list[ConversationInteraction]:
        return self.coordinator.session_memory.get_history(session_id)

    def retrieve_similar_fraud(self, session_id: str, transcript: str) -> dict[str, Any]:
        return self.coordinator.retrieve_similar_fraud(session_id=session_id, transcript=transcript)

    def enrich_risk(
        self,
        session_id: str,
        base_risk: dict[str, Any],
        retrieval_context: dict[str, Any],
    ) -> dict[str, Any]:
        return self.coordinator.enrich_risk(
            session_id=session_id,
            base_risk=base_risk,
            retrieval_context=retrieval_context,
        )
