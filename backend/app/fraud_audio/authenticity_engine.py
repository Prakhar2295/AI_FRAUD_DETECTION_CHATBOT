"""AuthenticityEngine orchestrates detectors and returns structured intelligence."""
from __future__ import annotations

from typing import Dict, Any
import asyncio
import numpy as np

from .replay_detector import ReplayDetector
from .spoof_detector import SpoofDetector
from .deepfake_detector import DeepfakeDetector
from .artifact_analyzer import analyze_artifacts
from .voice_risk_aggregator import aggregate_signals


class AuthenticityEngine:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.loop = loop or asyncio.get_event_loop()
        self.replay = ReplayDetector()
        self.spoof = SpoofDetector()
        self.deepfake = DeepfakeDetector()

    async def analyze(self, pcm: np.ndarray, sr: int) -> Dict[str, Any]:
        # Run lightweight detectors concurrently to avoid blocking
        tasks = [
            self.loop.run_in_executor(None, self.replay.analyze, pcm, sr),
            self.loop.run_in_executor(None, self.spoof.analyze, pcm, sr),
            self.loop.run_in_executor(None, self.deepfake.analyze, pcm, sr),
            self.loop.run_in_executor(None, analyze_artifacts, pcm),
        ]
        replay_res, spoof_res, deepfake_res, artifact_res = await asyncio.gather(*tasks)

        signals = {
            "replay": float(getattr(replay_res, "replay_confidence", 0.0)),
            "spoof": float(getattr(spoof_res, "spoof_confidence", 0.0)),
            "deepfake": float(getattr(deepfake_res, "deepfake_confidence", 0.0)),
        }

        agg = aggregate_signals(signals)

        result = {
            "authenticity_score": agg.authenticity_score,
            "components": agg.components,
            "metadata": {
                "replay": getattr(replay_res, "details", {}),
                "spoof": getattr(spoof_res, "details", {}),
                "deepfake": getattr(deepfake_res, "details", {}),
                "artifacts": artifact_res.details if hasattr(artifact_res, 'details') else {},
            },
        }

        return result
