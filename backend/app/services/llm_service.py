"""Ollama API client service."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import requests

from app.utils.logger import get_logger


class OllamaLLMService:
    """Reusable client for local Ollama text generation."""

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout_seconds: int = 120,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(self.__class__.__name__)

    def generate_response(self, prompt: str) -> dict[str, Any]:
        """Send a prompt to Ollama and parse the JSON object returned by the model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        try:
            self.logger.info("Sending prompt to Ollama model=%s", self.model_name)
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            raw_model_response = data.get("response", "")

            if not raw_model_response:
                raise ValueError("Ollama response did not include a 'response' field")

            return json.loads(raw_model_response)
        except requests.Timeout as exc:
            self.logger.error("Ollama request timed out after %ss", self.timeout_seconds)
            raise RuntimeError("Ollama request timed out") from exc
        except requests.RequestException as exc:
            response_text = getattr(exc.response, "text", "") if exc.response else ""
            detail = f"{exc}. Response body: {response_text}" if response_text else str(exc)
            self.logger.error("Ollama request failed: %s", detail)
            raise RuntimeError(f"Ollama request failed: {detail}") from exc
        except json.JSONDecodeError as exc:
            self.logger.error("Ollama returned invalid JSON")
            raise RuntimeError("Ollama returned invalid JSON") from exc

    async def generate_response_async(self, prompt: str) -> dict[str, Any]:
        """Async-ready wrapper for Ollama generation."""
        return await asyncio.to_thread(self.generate_response, prompt)
