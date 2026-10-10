from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

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
5. {"action":"run_command","args":{"command":"single command without pipes/redirection","cwd":"relative/project/directory"},"reason":"..."}
6. {"action":"finish","args":{"summary":"what was completed","verification":"what proves it works"},"reason":"..."}

Rules:
- Inspect before making assumptions about an existing project.
- Treat the latest user guidance as higher priority than the original implementation plan when they conflict.
- Prefer complete, production-quality edits over fragments.
- After editing code, run the relevant build/tests/linter when available.
- If a command or test fails, diagnose the observed error and fix it.
- NEVER use `cd some-dir && command`. Shell chaining is blocked. Instead set run_command.args.cwd to the target directory and put only the executable command in args.command.
- NEVER repeat an identical failing action. Change the action, arguments, cwd, or implementation based on the observed error.
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
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient,
        tools: WorkspaceTools,
        progress: Callable[[str], None] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        instruction_source: Callable[[], list[str]] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools
        self.progress = progress or (lambda _message: None)
        self.event_callback = event_callback or (lambda _event: None)
        self.instruction_source = instruction_source or (lambda: [])
        self.stop_requested = stop_requested or (lambda: False)

    def _event(
        self,
        kind: str,
        title: str,
        *,
        status: str = "info",
        detail: str = "",
        step: int | None = None,
        action: str | None = None,
        target: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self.event_callback({
            "kind": kind,
            "title": title,
            "status": status,
            "detail": detail,
            "step": step,
            "action": action,
            "target": target,
            "cwd": cwd,
            "max_steps": self.config.max_steps,
        })

    def plan(self, task: str) -> str:
        self.progress("[planner] Asking the model to create an implementation plan...")
        self._event("planner", "Creating implementation plan", status="running")
        try:
            plan = self.llm.complete(PLAN_PROMPT, task)
        except Exception as exc:
            self._event("planner", "Planning failed", status="error", detail=str(exc))
            raise
        self.progress("[planner] Plan ready.")
        self._event("planner", "Implementation plan ready", status="success", detail=plan)
        self._event("chat", "Agent", status="info", detail=f"I created this plan:\n\n{plan}")
        return plan

    def run(self, task: str, plan: str | None = None) -> AgentResult:
        plan = plan or self.plan(task)
        history: list[dict[str, Any]] = []
        guidance: list[str] = []
        initial_listing = self.tools.list_files(".")
        self.progress(f"[workspace] Initial files: {self._shorten(initial_listing, 500)}")
        self._event(
            "workspace",
            "Workspace inspected",
            status="success",
            detail=self._shorten(initial_listing, 2000),
            cwd=".",
        )

        for step in range(1, self.config.max_steps + 1):
            if self.stop_requested():
                self.progress("[agent] Stop requested by user.")
                self._event(
                    "finish",
                    "Agent stopped by user",
                    status="error",
                    detail="Stopped safely before the next reasoning step.",
                    step=step,
                )
                return AgentResult(False, "Stopped by user.", "No further actions were executed after the stop request.", step - 1, history)

            new_guidance = self.instruction_source()
            if new_guidance:
                guidance.extend(new_guidance)
                joined = "\n".join(f"- {item}" for item in new_guidance)
                self.progress(f"[user guidance]\n{joined}")
                self._event("chat", "Agent", status="info", detail="Got it. I will apply your latest instruction before continuing.")
                self._event("guidance", "Applying new user guidance", status="warning", detail=joined, step=step)
                history.append({"step": step, "action": "user_guidance", "observation": joined})

            self.progress(f"\n[step {step}/{self.config.max_steps}] Thinking about the next action...")
            self._event("thinking", "AI is deciding the next action", status="running", step=step)
            prompt = self._build_turn_prompt(task, plan, initial_listing, history, guidance, step)
            try:
                raw = self.llm.complete(SYSTEM_PROMPT, prompt)
            except Exception as exc:
                self._event("model", "Model request failed", status="error", detail=str(exc), step=step)
                raise

            try:
                decision = self._parse_decision(raw)
            except ValueError as exc:
                self.progress(f"[step {step}] Model returned invalid action JSON: {exc}")
                self._event("model", "Invalid model response; retrying", status="warning", detail=str(exc), step=step)
                history.append({"step": step, "action": "invalid_model_output", "observation": str(exc), "raw": raw[:2000]})
                continue

            action = decision["action"]
            args = decision.get("args", {})
            reason = decision.get("reason", "")
            target = self._action_target(action, args)
            cwd = str(args.get("cwd", ".")) if action == "run_command" else None

            repeated = self._repeated_failure_count(history, action, args)
            if repeated >= 2:
                observation = (
                    "REPEATED_ACTION_BLOCKED: This exact action already failed twice. "
                    "Do not repeat it. Change the command, cwd, file, or implementation based on the previous error."
                )
                self.progress(f"[step {step}] {observation}")
                self._event(
                    "loop",
                    "Repeated failing action blocked",
                    status="warning",
                    detail=observation,
                    step=step,
                    action=action,
                    target=target,
                    cwd=cwd,
                )
                history.append({
                    "step": step,
                    "action": action,
                    "args": self._safe_args_for_history(action, args),
                    "reason": reason,
                    "observation": observation,
                })
                continue

            self.progress(f"[step {step}] Action: {action}")
            if reason:
                self.progress(f"[step {step}] Why   : {reason}")
            self._report_action_details(step, action, args)
            self._event(
                "action",
                self._action_title(action, target, cwd),
                status="running",
                detail=reason,
                step=step,
                action=action,
                target=target,
                cwd=cwd,
            )

            if action == "finish":
                summary = str(args.get("summary", "Task completed."))
                verification = str(args.get("verification", ""))
                self.progress(f"[step {step}] Agent marked the task complete.")
                if verification:
                    self.progress(f"[verify] {verification}")
                self._event("chat", "Agent", status="info", detail=f"Completed.\n\n{summary}\n\nVerification: {verification}".strip())
                self._event(
                    "finish",
                    "Agent marked task complete",
                    status="success",
                    detail=f"{summary}\n\nVerification: {verification}".strip(),
                    step=step,
                    action=action,
                )
                history.append({"step": step, "action": action, "reason": reason})
                return AgentResult(True, summary, verification, step, history)

            self.progress(f"[step {step}] Executing {action}...")
            try:
                observation = self._execute(action, args)
                if action == "run_command" and "exit_code=0" not in observation:
                    event_status = "error"
                    event_title = f"{action} returned a non-zero exit code"
                else:
                    event_status = "success"
                    event_title = f"{action} completed"
            except (ToolError, ValueError, TypeError) as exc:
                observation = f"TOOL_ERROR: {exc}"
                event_status = "error"
                event_title = f"{action} failed"

            self.progress(f"[step {step}] Result : {self._shorten(observation, 1200)}")
            self._event(
                "result",
                event_title,
                status=event_status,
                detail=self._shorten(observation, 4000),
                step=step,
                action=action,
                target=target,
                cwd=cwd,
            )
            history.append({
                "step": step,
                "action": action,
                "args": self._safe_args_for_history(action, args),
                "reason": reason,
                "observation": observation,
            })

        self.progress(f"[agent] Reached maximum step limit: {self.config.max_steps}")
        self._event("finish", "Maximum step limit reached", status="error", detail=f"Stopped after {self.config.max_steps} steps.", step=self.config.max_steps)
        return AgentResult(False, f"Stopped after reaching the configured step limit ({self.config.max_steps}).", "Review the final observations and increase AGENT_MAX_STEPS if appropriate.", self.config.max_steps, history)

    @staticmethod
    def _repeated_failure_count(history: list[dict[str, Any]], action: str, args: dict[str, Any]) -> int:
        safe_args = AutonomousDeveloper._safe_args_for_history(action, args)
        count = 0
        for item in reversed(history[-8:]):
            if item.get("action") != action or item.get("args") != safe_args:
                continue
            observation = str(item.get("observation", ""))
            if "TOOL_ERROR:" in observation or "exit_code=" in observation and "exit_code=0" not in observation or "REPEATED_ACTION_BLOCKED" in observation:
                count += 1
        return count

    @staticmethod
    def _action_target(action: str, args: dict[str, Any]) -> str | None:
        if action in {"read_file", "write_file", "make_directory", "list_files"}:
            return str(args.get("path", "."))
        if action == "run_command":
            return str(args.get("command", ""))
        return None

    @staticmethod
    def _action_title(action: str, target: str | None, cwd: str | None = None) -> str:
        labels = {
            "list_files": "Inspecting files",
            "read_file": "Reading file",
            "write_file": "Writing file",
            "make_directory": "Creating directory",
            "run_command": "Running command",
            "finish": "Finishing task",
        }
        title = labels.get(action, action)
        if action == "run_command" and target:
            return f"{title} in {cwd or '.'}: {target}"
        return f"{title}: {target}" if target else title

    def _report_action_details(self, step: int, action: str, args: dict[str, Any]) -> None:
        if action in {"read_file", "write_file", "make_directory", "list_files"}:
            self.progress(f"[step {step}] Target: {args.get('path', '.')}")
        elif action == "run_command":
            self.progress(f"[step {step}] Working directory: {args.get('cwd', '.')}")
            self.progress(f"[step {step}] Command: {args.get('command', '')}")
        if action == "write_file":
            self.progress(f"[step {step}] Writing {len(str(args.get('content', '')))} characters.")

    def _execute(self, action: str, args: dict[str, Any]) -> str:
        if action == "list_files":
            return self.tools.list_files(str(args.get("path", ".")))
        if action == "read_file":
            return self.tools.read_file(self._required_string(args, "path"))
        if action == "write_file":
            return self.tools.write_file(self._required_string(args, "path"), self._required_string(args, "content", allow_empty=True))
        if action == "make_directory":
            return self.tools.make_directory(self._required_string(args, "path"))
        if action == "run_command":
            cwd = str(args.get("cwd", "."))
            return self.tools.run_command(self._required_string(args, "command"), cwd=cwd)
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
        if decision.get("action") not in {"list_files", "read_file", "write_file", "make_directory", "run_command", "finish"}:
            raise ValueError(f"Unsupported model action: {decision.get('action')!r}")
        if "args" in decision and not isinstance(decision["args"], dict):
            raise ValueError("'args' must be a JSON object.")
        return decision

    @staticmethod
    def _safe_args_for_history(action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action != "write_file":
            return args
        content = str(args.get("content", ""))
        return {"path": args.get("path"), "content_preview": content[:500], "content_length": len(content)}

    @staticmethod
    def _shorten(value: Any, limit: int) -> str:
        text = str(value).strip()
        return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"

    @staticmethod
    def _build_turn_prompt(
        task: str,
        plan: str,
        initial_listing: str,
        history: list[dict[str, Any]],
        guidance: list[str],
        step: int,
    ) -> str:
        recent_history = history[-12:]
        guidance_text = "\n".join(f"- {item}" for item in guidance[-10:]) or "No additional user guidance."
        return (
            f"USER TASK:\n{task}\n\n"
            f"IMPLEMENTATION PLAN:\n{plan}\n\n"
            f"LATEST USER GUIDANCE (highest priority when it conflicts with the plan):\n{guidance_text}\n\n"
            f"INITIAL WORKSPACE:\n{initial_listing}\n\n"
            f"CURRENT STEP: {step}\n\n"
            f"RECENT ACTIONS AND OBSERVATIONS:\n{json.dumps(recent_history, indent=2, ensure_ascii=False)}\n\n"
            "Choose the single best next action. Return JSON only."
        )
