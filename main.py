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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = args.task or input("Task > ").strip()
    if not task:
        print("A task is required.", file=sys.stderr)
        return 2

    try:
        config = AgentConfig.from_env(args.workspace)
        llm = build_llm(config)
        tools = WorkspaceTools(config.workspace, config.command_timeout)
        agent = AutonomousDeveloper(config, llm, tools)

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
        return 0 if result.completed else 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
