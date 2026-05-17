"""Structured fraud knowledge base for pattern enrichment."""
from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger


class FraudKnowledgeBase:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.entries: list[dict[str, Any]] = []
        self._load_default_entries()

    def _load_default_entries(self) -> None:
        self.entries = [
            {
                "key": "replay_attack",
                "title": "Voice replay fraud pattern",
                "description": "Attackers replay previously captured audio to impersonate the customer.",
                "tags": ["authenticity", "replay", "deepfake"],
            },
            {
                "key": "phishing_payment_change",
                "title": "Payment redirection social engineering",
                "description": "A scam targets payment account change under the guise of urgency.",
                "tags": ["social_engineering", "urgency"],
            },
        ]
        self.logger.info("Loaded fraud knowledge base entries=%s", len(self.entries))

    def query(self, tag: str | None = None) -> list[dict[str, Any]]:
        if tag is None:
            return [entry.copy() for entry in self.entries]
        return [entry.copy() for entry in self.entries if tag in entry.get("tags", [])]
