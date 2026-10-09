from __future__ import annotations

import json
import os
from typing import Protocol

import requests
from openai import OpenAI

from agent_config import AgentConfig


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class OpenAIResponsesClient:
    def __init__(self, config: AgentConfig) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when AGENT_PROVIDER=openai.")
        self.client = OpenAI()
        self.model = config.openai_model

    def complete(self, system: str, user: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
        )
        return response.output_text.strip()


class OllamaClient:
    def __init__(self, config: AgentConfig) -> None:
        self.url = f"{config.ollama_base_url}/api/chat"
        self.model = config.ollama_model

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.1},
        }

        try:
            # Local coding models can take several minutes on the first request,
            # especially larger models such as qwen3-coder:30B.
            response = requests.post(self.url, json=payload, timeout=(10, 600))
        except requests.exceptions.ConnectTimeout as exc:
            raise RuntimeError(
                f"Timed out connecting to Ollama at {self.url}. "
                "Check OLLAMA_BASE_URL and confirm Ollama is running."
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            raise RuntimeError(
                f"Ollama model '{self.model}' did not finish within 600 seconds. "
                "Try a smaller model or increase the client timeout."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.url}. "
                "Confirm Ollama is running and OLLAMA_BASE_URL is correct."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:1000]
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} for model "
                f"'{self.model}': {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Ollama returned a non-JSON response: {response.text[:500]}"
            ) from exc

        content = data.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:1000]}")
        return content.strip()


def build_llm(config: AgentConfig) -> LLMClient:
    if config.provider == "ollama":
        return OllamaClient(config)
    return OpenAIResponsesClient(config)
