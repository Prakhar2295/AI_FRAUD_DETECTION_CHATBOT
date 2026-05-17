"""Async-safe behavioral fraud analysis pipeline coordinator."""
from __future__ import annotations

import asyncio
from typing import Any

from .behavioral_risk_aggregator import aggregate_behavioral_signals
from .conversational_signal_extractor import extract_conversational_signals
from .emotional_analyzer import EmotionalAnalyzer
from .urgency_detector import UrgencyDetector
from .manipulation_detector import ManipulationDetector
from .stress_analyzer import StressAnalyzer
from .hesitation_analyzer import HesitationAnalyzer
from .social_engineering_detector import SocialEngineeringDetector


class BehavioralPipeline:
    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None, max_queue: int = 64):
        self.loop = loop or asyncio.get_event_loop()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self.emotional = EmotionalAnalyzer()
        self.urgency = UrgencyDetector()
        self.manipulation = ManipulationDetector()
        self.stress = StressAnalyzer()
        self.hesitation = HesitationAnalyzer()
        self.social_engineering = SocialEngineeringDetector()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = self.loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue_transcript(self, session_id: str, transcript: str, metadata: dict[str, Any]) -> dict[str, Any]:
        fut: asyncio.Future = self.loop.create_future()
        await self.queue.put({
            "session_id": session_id,
            "transcript": transcript,
            "metadata": metadata,
            "future": fut,
        })
        await self.start()
        return await fut

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            fut = item["future"]
            try:
                result = await self.loop.run_in_executor(None, self._analyze, item["session_id"], item["transcript"], item["metadata"])
                if not fut.done():
                    fut.set_result(result)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self.queue.task_done()

    def _analyze(self, session_id: str, transcript: str, metadata: dict[str, Any]) -> dict[str, Any]:
        signals = extract_conversational_signals(transcript)
        emotional = self.emotional.analyze(transcript)
        urgency = self.urgency.analyze(transcript)
        manipulation = self.manipulation.analyze(transcript)
        stress = self.stress.analyze(transcript)
        hesitation = self.hesitation.analyze(transcript)
        social_engineering = self.social_engineering.analyze(transcript)

        aggregated = aggregate_behavioral_signals(
            {
                "emotional": emotional.emotional_risk_score,
                "urgency": urgency.urgency_score,
                "manipulation": manipulation.manipulation_confidence,
                "hesitation": hesitation.hesitation_score,
                "stress": stress.stress_score,
                "social_engineering": social_engineering.social_engineering_confidence,
            }
        )

        return {
            "session_id": session_id,
            "transcript": transcript,
            "behavioral_risk_score": aggregated.behavioral_risk_score,
            "urgency_score": aggregated.urgency_score,
            "emotional_risk_score": aggregated.emotional_risk_score,
            "manipulation_confidence": aggregated.manipulation_confidence,
            "hesitation_score": aggregated.hesitation_score,
            "stress_score": aggregated.stress_score,
            "social_engineering_confidence": aggregated.social_engineering_confidence,
            "behavioral_metadata": {
                "conversational_signals": signals.__dict__,
                "aggregator": aggregated.metadata,
                "details": {
                    "emotional": emotional.details,
                    "urgency": urgency.details,
                    "manipulation": manipulation.details,
                    "stress": stress.details,
                    "hesitation": hesitation.details,
                    "social_engineering": social_engineering.details,
                },
            },
        }
*** End Patch