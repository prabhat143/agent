from pathlib import Path

import pytest

from tools import ToolError, WorkspaceTools


def test_write_and_read_file(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write_file("src/app.txt", "hello")
    assert tools.read_file("src/app.txt") == "hello"


def test_path_escape_is_blocked(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    with pytest.raises(ToolError):
        tools.write_file("../escape.txt", "no")


def test_dangerous_command_is_blocked(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    with pytest.raises(ToolError):
        tools.run_command("rm -rf .")


def test_simple_command_runs_inside_workspace(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    output = tools.run_command("python -c print('ok')")
    assert "exit_code=0" in output
    assert "ok" in output
