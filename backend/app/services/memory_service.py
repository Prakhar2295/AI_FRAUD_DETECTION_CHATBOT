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

        transcript = interaction.transcript or ""

        intent = ""
        if interaction.intent:
            intent = interaction.intent.customer_intent

        transaction_type = ""
        if interaction.intent:
            transaction_type = interaction.intent.transaction_type

        fraud_indicators = []
        fraud_reasoning = ""

        if interaction.fraud:
            fraud_indicators = interaction.fraud.suspicious_indicators or []
            fraud_reasoning = interaction.fraud.llm_reasoning or ""

        risk_level = ""
        risk_score = ""

        if interaction.risk:
            risk_level = interaction.risk.risk_level
            risk_score = interaction.risk.fraud_risk_score

        urgency_score = ""
        emotional_score = ""

        if interaction.behavioral:
            urgency_score = interaction.behavioral.urgency_score
            emotional_score = interaction.behavioral.emotional_risk_score

        embedding_text = f"""
        Transcript:
        {transcript}

        Intent:
        {intent}

        Transaction Type:
        {transaction_type}

        Fraud Indicators:
        {", ".join(fraud_indicators)}

        Fraud Reasoning:
        {fraud_reasoning}

        Risk Level:
        {risk_level}

        Risk Score:
        {risk_score}

        Urgency Score:
        {urgency_score}

        Emotional Risk Score:
        {emotional_score}
        """.strip()

        MAX_EMBEDDING_TEXT_LENGTH = 1500

        return embedding_text[:MAX_EMBEDDING_TEXT_LENGTH]
