"""Fuse transcript-based behavioral signals with audio emotion intelligence."""
from __future__ import annotations

from typing import Dict, Any
import logging

logger = logging.getLogger("BehavioralFusionService")


class BehavioralFusionService:
    def __init__(self, text_weight: float = 0.6, audio_weight: float = 0.4):
        self.text_weight = float(text_weight)
        self.audio_weight = float(audio_weight)

    def fuse(self, text_signals: Dict[str, Any], audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return fused behavioral assessment combining text and audio signals.

        text_signals: expected keys like urgency_score, manipulation_confidence, emotional_risk_score, stress_score
        audio_analysis: expected structure from AudioEmotionService
        """
        # Extract numeric text signals with safe defaults
        urgency = float(text_signals.get("urgency_score", 0.0) or 0.0)
        manipulation = float(text_signals.get("manipulation_confidence", 0.0) or 0.0)
        text_emotional = float(text_signals.get("emotional_risk_score", 0.0) or 0.0)
        text_stress = float(text_signals.get("stress_score", 0.0) or 0.0)

        # Audio-derived
        audio_probs = audio_analysis.get("emotion_probabilities", {}) if audio_analysis else {}
        dominant = audio_analysis.get("dominant_emotion") if audio_analysis else None
        audio_conf = float(audio_analysis.get("confidence", 0.0) or 0.0)

        # Simple audio risk estimates: map anger/fear/stress to scores
        audio_risk = 0.0
        if dominant:
            d = dominant.lower()
            if d == "angry":
                audio_risk = 0.9 * audio_conf
            elif d in ("fear", "fearful"):
                audio_risk = 0.8 * audio_conf
            elif d == "sad":
                audio_risk = 0.5 * audio_conf
            elif d == "neutral":
                audio_risk = 0.0
            elif d == "happy":
                audio_risk = 0.1 * audio_conf
            else:
                audio_risk = 0.2 * audio_conf

        # Weighted fusion for final behavioral risk
        text_component = (urgency * 0.5) + (manipulation * 0.4) + (text_emotional * 0.6)
        audio_component = audio_risk

        # normalize components to 0..1 scale (clamp)
        text_score = min(max(text_component / 2.0, 0.0), 1.0)
        audio_score = min(max(audio_component, 0.0), 1.0)

        behavioral_risk_score = (self.text_weight * text_score) + (self.audio_weight * audio_score)

        # Derive sub-scores
        urgency_score = min(max(urgency, 0.0), 1.0)
        emotional_risk_score = min(max((text_emotional + audio_score) / 2.0, 0.0), 1.0)
        stress_score = min(max((text_stress + audio_score) / 2.0, 0.0), 1.0)
        social_conf = min(max(text_signals.get("social_engineering_confidence", 0.0) if isinstance(text_signals, dict) else 0.0, 0.0), 1.0)

        fusion_metadata = {
            "text_risk_weight": self.text_weight,
            "audio_risk_weight": self.audio_weight,
            "fusion_strategy": "weighted_behavioral_fusion",
        }

        out = {
            # scale behavioral_risk_score to 0-100 to match existing response model
            "behavioral_risk_score": int(min(max(behavioral_risk_score * 100.0, 0), 100)),
            "urgency_score": float(urgency_score),
            "emotional_risk_score": float(emotional_risk_score),
            "stress_score": float(stress_score),
            "social_engineering_confidence": float(social_conf),
            "audio_emotion": {"dominant_emotion": dominant or "unknown", "confidence": float(audio_conf)},
            "fusion_metadata": fusion_metadata,
        }

        logger.info("Behavioral fusion output: %s", out)
        return out
