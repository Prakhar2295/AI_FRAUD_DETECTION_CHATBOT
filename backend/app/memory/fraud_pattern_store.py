"""In-memory fraud pattern registry and archetype store."""
from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger


class FraudPatternStore:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._patterns: list[dict[str, Any]] = []
        self._load_default_patterns()

    def _load_default_patterns(self) -> None:
        self._patterns = [
            {
                "pattern_id": "urgent_account_update",
                "name": "Urgent account update scam",
                "tags": ["urgency", "social_engineering", "account_info"],
                "description": "A fraudster pressures the victim to update account details immediately.",
                "example": "We need that account number right now or your access will be revoked.",
            },
            {
                "pattern_id": "technical_support_premise",
                "name": "Fake technical support",
                "tags": ["authority", "social_engineering", "refund"],
                "description": "The caller impersonates support staff and requests sensitive information.",
                "example": "This is the bank fraud team; please share your password to unlock your account.",
            },
            {
                "pattern_id": "payment_redirection",
                "name": "Payment redirection",
                "tags": ["transaction_change", "rush", "impersonation"],
                "description": "The caller asks the user to redirect a legitimate payment to a new account.",
                "example": "Transfer this payment to our secure code account to avoid delays.",
            },
        ]
        self.logger.info("Loaded default fraud patterns count=%s", len(self._patterns))

    def register_pattern(self, pattern: dict[str, Any]) -> None:
        self._patterns.append(pattern)
        self.logger.info("Registered new fraud pattern id=%s", pattern.get("pattern_id"))

    def get_patterns(self) -> list[dict[str, Any]]:
        return [pattern.copy() for pattern in self._patterns]

    def find_matching_patterns(self, text: str, threshold: float = 0.25) -> list[dict[str, Any]]:
        normalized = text.lower()
        matches: list[dict[str, Any]] = []
        for pattern in self._patterns:
            score = 0.0
            example = pattern.get("example", "").lower()
            description = pattern.get("description", "").lower()
            if example and example in normalized:
                score += 0.9
            if any(tag in normalized for tag in pattern.get("tags", [])):
                score += 0.2
            if description and any(word in normalized for word in description.split()[:4]):
                score += 0.1
            if score >= threshold:
                matches.append({**pattern, "match_score": round(score, 2)})
        return matches
