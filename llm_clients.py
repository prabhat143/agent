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
            response = requests.post(self.url, json=payload, timeout=180)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama. Make sure Ollama is running and the configured model is installed."
            ) from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
        return content.strip()


def build_llm(config: AgentConfig) -> LLMClient:
    if config.provider == "ollama":
        return OllamaClient(config)
    return OpenAIResponsesClient(config)
