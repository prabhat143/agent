from __future__ import annotations

import argparse
import sys

from agent_config import AgentConfig
from llm_clients import build_llm
from orchestrator import AutonomousDeveloper
from tools import WorkspaceTools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTA Autonomous Developer Agent")
    parser.add_argument("task", nargs="?", help="Development task to complete")
    parser.add_argument("--workspace", help="Workspace directory the agent may modify")
    parser.add_argument("--ui", action="store_true", help="Open the interactive chat dashboard")
    parser.add_argument("--ui-port", type=int, default=8765, help="Dashboard port (default: 8765)")
    return parser.parse_args()


def print_progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = parse_args()
    task = args.task or input("Task > ").strip()
    if not task:
        print("A task is required.", file=sys.stderr)
        return 2

    dashboard_store = None

    try:
        config = AgentConfig.from_env(args.workspace)
        llm = build_llm(config)
        tools = WorkspaceTools(config.workspace, config.command_timeout)

        event_callback = None
        instruction_source = None
        stop_requested = None
        if args.ui:
            from chat_dashboard import ChatProgressStore, start_chat_dashboard

            dashboard_store = ChatProgressStore()
            dashboard_store.set_context(task, config.provider, str(config.workspace))
            event_callback = dashboard_store.publish
            instruction_source = dashboard_store.drain_instructions
            stop_requested = dashboard_store.stop_requested
            start_chat_dashboard(dashboard_store, port=args.ui_port)
            print(f"Dashboard: http://127.0.0.1:{args.ui_port}")

        agent = AutonomousDeveloper(
            config,
            llm,
            tools,
            progress=print_progress,
            event_callback=event_callback,
            instruction_source=instruction_source,
            stop_requested=stop_requested,
        )

        print(f"Provider : {config.provider}")
        print(f"Workspace: {config.workspace}")
        print("\nPlanning...\n")
        plan = agent.plan(task)
        print(plan)
        print("\nExecuting...\n")

        result = agent.run(task, plan=plan)

        print("\n=== Result ===")
        print(f"Completed   : {result.completed}")
        print(f"Steps       : {result.steps}")
        print(f"Summary     : {result.summary}")
        if result.verification:
            print(f"Verification: {result.verification}")

        if dashboard_store is not None and not result.completed and not dashboard_store.stop_requested():
            dashboard_store.publish({
                "kind": "finish",
                "title": "Task stopped before successful completion",
                "status": "error",
                "detail": f"{result.summary}\n\n{result.verification}".strip(),
                "step": result.steps,
                "max_steps": config.max_steps,
            })

        return 0 if result.completed else 1
    except KeyboardInterrupt:
        if dashboard_store is not None:
            dashboard_store.publish({"kind": "finish", "title": "Agent interrupted", "status": "error", "detail": "The process was stopped by the user."})
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if dashboard_store is not None:
            dashboard_store.publish({"kind": "finish", "title": "Agent failed", "status": "error", "detail": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
