"""Registry of fraud archetypes and narrative templates."""
from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger


class FraudPatternRegistry:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._registry: dict[str, dict[str, Any]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        patterns = [
            {
                "pattern_id": "account_verification",
                "label": "Account verification pressure",
                "description": "Scammer requests account details using urgency and authority language.",
            },
            {
                "pattern_id": "callback_impersonation",
                "label": "False callback from bank staff",
                "description": "Scammer claims to return a prior call and requests immediate action.",
            },
        ]
        for pattern in patterns:
            self._registry[pattern["pattern_id"]] = pattern
        self.logger.info("Initialized fraud pattern registry with %s patterns", len(self._registry))

    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None:
        return self._registry.get(pattern_id)

    def list_patterns(self) -> list[dict[str, Any]]:
        return [pattern.copy() for pattern in self._registry.values()]
