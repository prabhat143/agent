from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    openai_model: str
    ollama_model: str
    ollama_base_url: str
    max_steps: int
    command_timeout: int
    workspace: Path

    @classmethod
    def from_env(cls, workspace_override: str | None = None) -> "AgentConfig":
        provider = os.getenv("AGENT_PROVIDER", "openai").strip().lower()
        if provider not in {"openai", "ollama"}:
            raise ValueError("AGENT_PROVIDER must be either 'openai' or 'ollama'.")

        workspace_value = workspace_override or os.getenv("AGENT_WORKSPACE", "workspace")
        workspace = Path(workspace_value).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        return cls(
            provider=provider,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.3-codex"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3-coder"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            max_steps=max(1, int(os.getenv("AGENT_MAX_STEPS", "30"))),
            command_timeout=max(1, int(os.getenv("AGENT_COMMAND_TIMEOUT", "120"))),
            workspace=workspace,
        )
