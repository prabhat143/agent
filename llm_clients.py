from __future__ import annotations

import json
import os
from typing import Protocol

import requests

from agent_config import AgentConfig


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class OpenAIResponsesClient:
    def __init__(self, config: AgentConfig) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when AGENT_PROVIDER=openai.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support is optional. Install it with: pip install openai"
            ) from exc

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

    @staticmethod
    def _requires_json(system: str) -> bool:
        text = system.lower()
        return (
            "return only a json object" in text
            or "return json only" in text
            or "exactly one tool action" in text
        )

    @staticmethod
    def _decision_policy() -> str:
        return """

DECISION POLICY FOR SOFTWARE FIXES:
- Fix the smallest concrete observed error first. Do not broaden the investigation unless that fix fails.
- Treat compiler/test output as the primary source of truth.
- For Java `cannot find symbol`, package, import, method-name, or type errors: inspect the failing source file and the referenced class declaration, make the smallest source edit (often an import/package/method correction), then rerun the exact failed Maven command immediately.
- Do NOT run `mvn dependency:tree` unless the observed error actually mentions dependency resolution, a missing artifact, incompatible version, or classpath conflict.
- Do NOT inspect or create application.properties/application.yml unless the observed error points to configuration, datasource, startup, profile, or property binding.
- Do NOT inspect unrelated controllers/services/repositories when the compiler already identifies the failing file and symbol.
- After at most two focused read_file actions for the same compile error, choose write_file and apply the minimal fix, unless the required information is genuinely unavailable.
- After write_file for a build/test failure, rerun the same failed build/test command before doing any broader diagnostics.
- Prefer a one-line import/package/method fix over creating new files, changing dependencies, or restructuring packages when that resolves the observed error.
- `write_file` creates a file if it does not exist and replaces it if it does. Never invent `create_file`, `edit_file`, or `update_file`; use `write_file`.
"""

    @staticmethod
    def _normalize_tool_decision(content: str) -> str:
        """Normalize common local-model action aliases before orchestration validation."""
        try:
            decision = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return content

        if not isinstance(decision, dict):
            return content

        aliases = {
            "create_file": "write_file",
            "edit_file": "write_file",
            "update_file": "write_file",
            "mkdir": "make_directory",
            "shell": "run_command",
            "execute_command": "run_command",
        }
        action = decision.get("action")
        if isinstance(action, str) and action in aliases:
            decision["action"] = aliases[action]

        return json.dumps(decision, ensure_ascii=False)

    def complete(self, system: str, user: str) -> str:
        requires_json = self._requires_json(system)
        effective_system = system
        if requires_json:
            effective_system += self._decision_policy()

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.0 if requires_json else 0.1},
        }

        # Plans and chat can remain natural language, but autonomous tool decisions
        # must be machine-readable. Ollama's JSON mode significantly reduces cases
        # where a local model returns prose or a JSON object without the expected
        # action envelope.
        if requires_json:
            payload["format"] = "json"

        try:
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

        content = content.strip()
        if requires_json:
            content = self._normalize_tool_decision(content)
        return content


def build_llm(config: AgentConfig) -> LLMClient:
    if config.provider == "ollama":
        return OllamaClient(config)
    return OpenAIResponsesClient(config)
