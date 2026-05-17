"""Memory orchestration layer for fraud persistence and retrieval."""
from __future__ import annotations

from typing import Any

from app.memory.adaptive_risk_engine import AdaptiveRiskEngine
from app.memory.fraud_pattern_store import FraudPatternStore
from app.memory.memory_pipeline import MemoryPipeline
from app.memory.retrieval_engine import RetrievalEngine
from app.memory.session_memory import SessionMemory
from app.vectorstore.chroma_manager import ChromaManager
from app.vectorstore.embedding_service import EmbeddingService
from app.utils.logger import get_logger


class MemoryCoordinator:
    def __init__(self) -> None:
        self.session_memory = SessionMemory()
        self.embedding_service = EmbeddingService()
        self.vector_store = ChromaManager()
        self.pattern_store = FraudPatternStore()
        self.retrieval_engine = RetrievalEngine(
            vector_store=self.vector_store,
            pattern_store=self.pattern_store,
            embedding_service=self.embedding_service,
        )
        self.adaptive_risk_engine = AdaptiveRiskEngine(session_memory=self.session_memory)
        self.pipeline = MemoryPipeline()
        self.logger = get_logger(self.__class__.__name__)

    async def start(self) -> None:
        await self.pipeline.start()

    async def stop(self) -> None:
        await self.pipeline.stop()

    async def persist_interaction(
        self,
        session_id: str,
        interaction: Any,
        embedding_text: str | None = None,
    ) -> None:
        async def _persist() -> None:
            self.persist_interaction_sync(session_id, interaction, embedding_text)

        await self.pipeline.enqueue({"task": "persist_interaction", "handler": _persist})

    def persist_interaction_sync(
        self,
        session_id: str,
        interaction: Any,
        embedding_text: str | None = None,
    ) -> None:
        session = self.session_memory.append_interaction(session_id, interaction)
        content = embedding_text or interaction.transcript
        if not content:
            return

        document_id = f"{session_id}-{len(session.interactions)}"
        embedding = self.embedding_service.embed_text(content)
        metadata = self._build_metadata(session_id, interaction)
        self.vector_store.upsert_document(
            document_id=document_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
        )
        self.logger.info("Persisted fraud memory document_id=%s", document_id)

    @staticmethod
    def _model_to_dict(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value

    def _build_metadata(self, session_id: str, interaction: Any) -> dict[str, Any]:
        risk = self._model_to_dict(getattr(interaction, "risk", None))
        fraud = self._model_to_dict(getattr(interaction, "fraud", None))
        behavioral = self._model_to_dict(getattr(interaction, "behavioral", None))
        return {
            "session_id": session_id,
            "timestamp": getattr(interaction, "timestamp", ""),
            "transcript": getattr(interaction, "transcript", ""),
            "risk_level": (risk or {}).get("risk_level") if isinstance(risk, dict) else None,
            "fraud_risk_score": (risk or {}).get("fraud_risk_score") if isinstance(risk, dict) else None,
            "suspicious_indicators": ", ".join((fraud or {}).get("suspicious_indicators", [])) if isinstance(fraud, dict) else "",
            "behavioral_risk_score": (behavioral or {}).get("behavioral_risk_score") if isinstance(behavioral, dict) else None,
        }

    def retrieve_similar_fraud(self, session_id: str, transcript: str) -> dict[str, Any]:
        retrieved = self.retrieval_engine.retrieve_similar_fraud(transcript)
        historical_context = {
            "session_summary": self.session_memory.get_session_summary(session_id),
            "recent_transcripts": self.session_memory.get_recent_transcripts(session_id),
        }
        return {
            **retrieved,
            "historical_fraud_context": historical_context,
        }

    def enrich_risk(self, session_id: str, base_risk: dict[str, Any], retrieval_context: dict[str, Any]) -> dict[str, Any]:
        return self.adaptive_risk_engine.enrich_risk(
            session_id=session_id,
            base_risk=base_risk,
            retrieval_context=retrieval_context,
        )
