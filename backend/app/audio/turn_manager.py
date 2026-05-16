"""Turn management for conversational speaker state."""

from __future__ import annotations


class TurnManager:
    """Track deterministic conversational turn progress."""

    def __init__(self) -> None:
        self.active_speaker: str = "none"
        self.turn_count: int = 0
        self.user_turn_active: bool = False
        self.ai_turn_active: bool = False

    def start_user_turn(self) -> None:
        self.user_turn_active = True
        self.ai_turn_active = False
        self.active_speaker = "user"

    def complete_user_turn(self) -> None:
        self.user_turn_active = False
        self.active_speaker = "none"

    def start_ai_turn(self) -> None:
        self.ai_turn_active = True
        self.user_turn_active = False
        self.turn_count += 1
        self.active_speaker = "assistant"

    def complete_ai_turn(self) -> None:
        self.ai_turn_active = False
        self.active_speaker = "none"

    def snapshot(self) -> dict[str, object]:
        return {
            "active_speaker": self.active_speaker,
            "turn_count": self.turn_count,
            "user_turn_active": self.user_turn_active,
            "ai_turn_active": self.ai_turn_active,
        }
