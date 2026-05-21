"""Behavioral fraud intelligence workflow node."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.state import FraudWorkflowState, append_trace
from app.models.response_models import BehavioralAnalysisResponse
from app.services.llm_service import OllamaLLMService
from app.services.behavioral_fusion_service import BehavioralFusionService
from app.utils.logger import get_logger

logger = get_logger("BehavioralNode")


def create_behavioral_node(llm_service: OllamaLLMService):
    """Create a LangGraph node that enriches behavioral fraud intelligence."""

    def behavioral_node(state: FraudWorkflowState) -> dict[str, Any]:
        logger.info("Executing behavioral node")
        prompt = _build_prompt(state)

        fusion_service = BehavioralFusionService()
        updates = append_trace(state, "behavioral_node")
        try:
            payload = llm_service.generate_response(prompt)
            # Validate LLM output into canonical behavioral model
            behavioral_model = BehavioralAnalysisResponse(**payload)
            behavioral_dict = _model_to_dict(behavioral_model)

            # Build text signals from LLM output and any pipeline metadata
            text_signals: dict[str, Any] = {}
            # copy numeric signals if present
            for key in ("urgency_score", "manipulation_confidence", "emotional_risk_score", "stress_score", "social_engineering_confidence", "hesitation_score"):
                text_signals[key] = behavioral_dict.get(key, 0.0)

            # merge deterministic behavioral metadata if present
            bm = state.get("behavioral_metadata") or {}
            if isinstance(bm, dict):
                agg = bm.get("aggregator") or bm
                for k in ("urgency_score", "manipulation_confidence", "emotional_risk_score", "stress_score", "social_engineering_confidence"):
                    if k not in text_signals or not text_signals.get(k):
                        text_signals[k] = agg.get(k, text_signals.get(k, 0.0))

            audio_analysis = state.get("audio_emotion_analysis")

            fused = fusion_service.fuse(text_signals, audio_analysis)

            # Build final BehavioralAnalysisResponse using fused values and LLM indicators
            final_behavioral = BehavioralAnalysisResponse(
                behavioral_risk_score=int(fused.get("behavioral_risk_score", 0)),
                urgency_score=float(fused.get("urgency_score", 0.0)),
                emotional_risk_score=float(fused.get("emotional_risk_score", 0.0)),
                manipulation_indicators=behavioral_dict.get("manipulation_indicators", []),
                hesitation_score=float(behavioral_dict.get("hesitation_score", 0.0)),
                stress_score=float(fused.get("stress_score", 0.0)),
                social_engineering_confidence=float(fused.get("social_engineering_confidence", 0.0)),
                metadata={"fusion": fused.get("fusion_metadata", {}), **(behavioral_dict.get("metadata") or {})},
            )

            return {
                **updates,
                "behavioral": _model_to_dict(final_behavioral),
            }
        except (RuntimeError, TypeError, ValidationError) as exc:
            logger.exception("Behavioral node LLM failed: %s", exc)
            # fallback: attempt deterministic fusion using existing behavioral_metadata and audio
            bm = state.get("behavioral_metadata") or {}
            text_signals = {}
            if isinstance(bm, dict):
                agg = bm.get("aggregator") or bm
                for k in ("urgency_score", "manipulation_confidence", "emotional_risk_score", "stress_score", "social_engineering_confidence"):
                    text_signals[k] = agg.get(k, 0.0)

            audio_analysis = state.get("audio_emotion_analysis")
            fused = fusion_service.fuse(text_signals, audio_analysis)
            fallback = BehavioralAnalysisResponse(
                behavioral_risk_score=int(fused.get("behavioral_risk_score", 0)),
                urgency_score=float(fused.get("urgency_score", 0.0)),
                emotional_risk_score=float(fused.get("emotional_risk_score", 0.0)),
                manipulation_indicators=[],
                hesitation_score=float(fused.get("hesitation_score", 0.0)),
                stress_score=float(fused.get("stress_score", 0.0)),
                social_engineering_confidence=float(fused.get("social_engineering_confidence", 0.0)),
                metadata={"fusion": fused.get("fusion_metadata", {}), "reason": "behavioral_node_failed"},
            )
            return {
                **updates,
                "behavioral": _model_to_dict(fallback),
                "errors": [*state.get("errors", []), f"behavioral_node: {exc}"],
            }

    return behavioral_node


def _build_prompt(state: FraudWorkflowState) -> str:

    transcript = state["transcript"]

    fraud_audio = state.get("fraud_audio") or {}

    behavioral_metadata = state.get("behavioral_metadata") or {}

    audio_emotion_analysis = state.get("audio_emotion_analysis") or {}

    return f"""
You are an expert Behavioral Fraud Intelligence System specialized in:

- banking fraud detection
- social engineering detection
- scam conversation analysis
- emotional manipulation analysis
- conversational behavioral intelligence

Your task is to analyze the transcript and detect behavioral fraud signals.

IMPORTANT RULES:

1. Detect conversational manipulation patterns.
2. Detect urgency pressure.
3. Detect emotional coercion.
4. Detect social engineering attempts.
5. Distinguish between:
   - genuine customer stress
   - scammer manipulation
6. Use transcript semantics + emotional context together.
7. Consider multilingual conversational style including:
   - Indian English
   - Hinglish
   - conversational Hindi-English patterns
8. Do NOT hallucinate fraud.
9. Neutral conversations must return low scores.
10. Scores must be between 0.0 and 1.0.

SCORING GUIDELINES:

0.0 → no signal
0.2 → weak signal
0.5 → moderate signal
0.8 → strong signal
1.0 → extreme signal

Behavioral Signals Definition:

- urgency_score:
  pressure for immediate action

- emotional_risk_score:
  emotional coercion / fear / panic

- hesitation_score:
  uncertainty or nervousness

- stress_score:
  vocal or conversational distress

- social_engineering_confidence:
  impersonation / manipulation / deception likelihood

- behavioral_risk_score:
  overall combined fraud behavioral risk

Few-shot Examples:

Example 1:
Transcript:
"Please share OTP immediately otherwise your account will be blocked."

Output:
{{
  "behavioral_risk_score": 0.91,
  "urgency_score": 0.95,
  "emotional_risk_score": 0.82,
  "manipulation_indicators": [
    "fear_pressure",
    "forced_urgency",
    "account_threat"
  ],
  "hesitation_score": 0.05,
  "stress_score": 0.71,
  "social_engineering_confidence": 0.93,
  "metadata": {{
    "summary": "Strong fraud manipulation using urgency and fear-based account blocking threats."
  }}
}}

Example 2:
Transcript:
"I think maybe there is some issue with my transaction."

Output:
{{
  "behavioral_risk_score": 0.18,
  "urgency_score": 0.10,
  "emotional_risk_score": 0.15,
  "manipulation_indicators": [],
  "hesitation_score": 0.42,
  "stress_score": 0.22,
  "social_engineering_confidence": 0.05,
  "metadata": {{
    "summary": "Customer appears uncertain but no fraud manipulation indicators detected."
  }}
}}

Example 3:
Transcript:
"This is the bank security department. Verify your account immediately."

Output:
{{
  "behavioral_risk_score": 0.87,
  "urgency_score": 0.83,
  "emotional_risk_score": 0.61,
  "manipulation_indicators": [
    "authority_impersonation",
    "forced_verification"
  ],
  "hesitation_score": 0.02,
  "stress_score": 0.44,
  "social_engineering_confidence": 0.95,
  "metadata": {{
    "summary": "Likely authority impersonation with forced account verification request."
  }}
}}

Return ONLY valid JSON.

JSON Schema:
{{
  "behavioral_risk_score": 0.0,
  "urgency_score": 0.0,
  "emotional_risk_score": 0.0,
  "manipulation_indicators": [],
  "hesitation_score": 0.0,
  "stress_score": 0.0,
  "social_engineering_confidence": 0.0,
  "metadata": {{
    "summary": ""
  }}
}}

Transcript:
{transcript}

Audio Emotion Analysis:
{audio_emotion_analysis}

Fraud Audio Metadata:
{fraud_audio}

Behavioral Metadata:
{behavioral_metadata}
""".strip()

def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
