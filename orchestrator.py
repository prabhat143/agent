from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent_config import AgentConfig
from llm_clients import LLMClient
from tools import ToolError, WorkspaceTools


SYSTEM_PROMPT = """You are an autonomous senior software engineer operating inside a restricted local workspace.
Your job is to complete the user's software task, not merely explain how to do it.

You may use exactly one tool action per turn. Return ONLY a JSON object, with no markdown.

Allowed actions:
1. {"action":"list_files","args":{"path":"."},"reason":"..."}
2. {"action":"read_file","args":{"path":"relative/path"},"reason":"..."}
3. {"action":"write_file","args":{"path":"relative/path","content":"complete file contents"},"reason":"..."}
4. {"action":"make_directory","args":{"path":"relative/path"},"reason":"..."}
5. {"action":"run_command","args":{"command":"single command without pipes/redirection"},"reason":"..."}
6. {"action":"finish","args":{"summary":"what was completed","verification":"what proves it works"},"reason":"..."}

Rules:
- Inspect before making assumptions about an existing project.
- Prefer complete, production-quality edits over fragments.
- After editing code, run the relevant build/tests/linter when available.
- If a command or test fails, diagnose the observed error and fix it.
- Do not claim success without verification when verification is possible.
- Stay inside the workspace.
- Never request secrets or embed API keys in source files.
- Avoid destructive commands.
- Keep working until acceptance criteria are met or you have a concrete blocker.
"""

PLAN_PROMPT = """Create a short implementation plan for this development task.
Return plain text with 3-8 concrete steps. Include how you will verify the result.
Do not write code yet.
"""


@dataclass
class AgentResult:
    completed: bool
    summary: str
    verification: str = ""
    steps: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class AutonomousDeveloper:
    def __init__(self, config: AgentConfig, llm: LLMClient, tools: WorkspaceTools) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools

    def plan(self, task: str) -> str:
        return self.llm.complete(PLAN_PROMPT, task)

    def run(self, task: str, plan: str | None = None) -> AgentResult:
        plan = plan or self.plan(task)
        history: list[dict[str, Any]] = []
        initial_listing = self.tools.list_files(".")

        for step in range(1, self.config.max_steps + 1):
            prompt = self._build_turn_prompt(task, plan, initial_listing, history, step)
            raw = self.llm.complete(SYSTEM_PROMPT, prompt)

            try:
                decision = self._parse_decision(raw)
            except ValueError as exc:
                history.append({
                    "step": step,
                    "action": "invalid_model_output",
                    "observation": str(exc),
                    "raw": raw[:2000],
                })
                continue

            action = decision["action"]
            args = decision.get("args", {})
            reason = decision.get("reason", "")

            if action == "finish":
                summary = str(args.get("summary", "Task completed."))
                verification = str(args.get("verification", ""))
                history.append({"step": step, "action": action, "reason": reason})
                return AgentResult(True, summary, verification, step, history)

            try:
                observation = self._execute(action, args)
            except (ToolError, ValueError, TypeError) as exc:
                observation = f"TOOL_ERROR: {exc}"

            history.append(
                {
                    "step": step,
                    "action": action,
                    "args": self._safe_args_for_history(action, args),
                    "reason": reason,
                    "observation": observation,
                }
            )

        return AgentResult(
            completed=False,
            summary=f"Stopped after reaching the configured step limit ({self.config.max_steps}).",
            verification="Review the final observations and increase AGENT_MAX_STEPS if appropriate.",
            steps=self.config.max_steps,
            history=history,
        )

    def _execute(self, action: str, args: dict[str, Any]) -> str:
        if action == "list_files":
            return self.tools.list_files(str(args.get("path", ".")))
        if action == "read_file":
            return self.tools.read_file(self._required_string(args, "path"))
        if action == "write_file":
            return self.tools.write_file(
                self._required_string(args, "path"),
                self._required_string(args, "content", allow_empty=True),
            )
        if action == "make_directory":
            return self.tools.make_directory(self._required_string(args, "path"))
        if action == "run_command":
            return self.tools.run_command(self._required_string(args, "command"))
        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _required_string(args: dict[str, Any], key: str, allow_empty: bool = False) -> str:
        value = args.get(key)
        if not isinstance(value, str):
            raise ValueError(f"'{key}' must be a string.")
        if not allow_empty and not value.strip():
            raise ValueError(f"'{key}' cannot be empty.")
        return value

    @staticmethod
    def _parse_decision(raw: str) -> dict[str, Any]:
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        try:
            decision = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}") from exc

        if not isinstance(decision, dict):
            raise ValueError("Model response must be a JSON object.")
        if decision.get("action") not in {
            "list_files",
            "read_file",
            "write_file",
            "make_directory",
            "run_command",
            "finish",
        }:
            raise ValueError(f"Unsupported model action: {decision.get('action')!r}")
        if "args" in decision and not isinstance(decision["args"], dict):
            raise ValueError("'args' must be a JSON object.")
        return decision

    @staticmethod
    def _safe_args_for_history(action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action != "write_file":
            return args
        content = str(args.get("content", ""))
        return {
            "path": args.get("path"),
            "content_preview": content[:500],
            "content_length": len(content),
        }

    @staticmethod
    def _build_turn_prompt(
        task: str,
        plan: str,
        initial_listing: str,
        history: list[dict[str, Any]],
        step: int,
    ) -> str:
        recent_history = history[-12:]
        return (
            f"USER TASK:\n{task}\n\n"
            f"IMPLEMENTATION PLAN:\n{plan}\n\n"
            f"INITIAL WORKSPACE:\n{initial_listing}\n\n"
            f"CURRENT STEP: {step}\n\n"
            f"RECENT ACTIONS AND OBSERVATIONS:\n"
            f"{json.dumps(recent_history, indent=2, ensure_ascii=False)}\n\n"
            "Choose the single best next action. Return JSON only."
        )
