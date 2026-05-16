"""Banking fraud analysis service."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from app.models.response_models import FraudAnalysisResponse
from app.services.llm_service import OllamaLLMService
from app.utils.logger import get_logger


class FraudAnalysisService:
    """Analyze transcribed speech for early banking fraud risk signals."""

    def __init__(self, llm_service: OllamaLLMService) -> None:
        self.llm_service = llm_service
        self.logger = get_logger(self.__class__.__name__)

    def analyze_transcription(self, transcription: str) -> FraudAnalysisResponse:
        """Return a structured fraud risk assessment for the given transcription."""
        if not transcription.strip():
            raise ValueError("Transcription cannot be empty")

        prompt = self._build_prompt(transcription)
        llm_payload = self.llm_service.generate_response(prompt)

        try:
            response = FraudAnalysisResponse(
                transcription=transcription,
                fraud_risk_score=llm_payload["fraud_risk_score"],
                risk_level=llm_payload["risk_level"],
                suspicious_indicators=llm_payload["suspicious_indicators"],
                llm_reasoning=llm_payload["llm_reasoning"],
            )
            self.logger.info(
                "Fraud analysis completed: risk_level=%s score=%s",
                response.risk_level,
                response.fraud_risk_score,
            )
            return response
        except (KeyError, TypeError, ValidationError) as exc:
            self.logger.error("LLM response did not match fraud analysis schema: %s", exc)
            raise RuntimeError("Invalid fraud analysis response from LLM") from exc

    async def analyze_transcription_async(
        self,
        transcription: str,
    ) -> FraudAnalysisResponse:
        """Async-ready wrapper for fraud analysis."""
        return await asyncio.to_thread(self.analyze_transcription, transcription)

    @staticmethod
    def _build_prompt(transcription: str) -> str:
        """Create the banking fraud analysis prompt for the local LLM."""
        return f"""
You are a banking fraud detection analyst reviewing a transcribed voice interaction.

Analyze the transcript for:
- urgency manipulation
- suspicious intent
- emotional pressure patterns
- requests for OTP, PIN, CVV, passwords, account access, remote access, or immediate transfer
- impersonation of bank staff, law enforcement, government, courier, or support agent

Return only valid JSON with this exact schema:
{{
  "fraud_risk_score": 0,
  "risk_level": "low | medium | high",
  "suspicious_indicators": ["indicator 1", "indicator 2"],
  "llm_reasoning": "short explanation"
}}

Scoring guidance:
- 0-30: low risk
- 31-70: medium risk
- 71-100: high risk

Transcript:
\"\"\"{transcription}\"\"\"
""".strip()
