from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


class ToolError(RuntimeError):
    pass


class WorkspaceTools:
    def __init__(self, workspace: Path, command_timeout: int = 120) -> None:
        self.workspace = workspace.resolve()
        self.command_timeout = command_timeout
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.workspace / relative_path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError("Path escapes the configured workspace.") from exc
        return candidate

    def list_files(self, path: str = ".") -> str:
        root = self._resolve(path)
        if not root.exists():
            raise ToolError(f"Path does not exist: {path}")
        if root.is_file():
            return str(root.relative_to(self.workspace))

        items: list[str] = []
        for item in sorted(root.rglob("*")):
            if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in item.parts):
                continue
            rel = item.relative_to(self.workspace)
            items.append(f"{rel}/" if item.is_dir() else str(rel))
            if len(items) >= 500:
                items.append("... truncated at 500 entries ...")
                break
        return "\n".join(items) if items else "(workspace is empty)"

    def read_file(self, path: str) -> str:
        file_path = self._resolve(path)
        if not file_path.is_file():
            raise ToolError(f"File does not exist: {path}")
        data = file_path.read_text(encoding="utf-8")
        if len(data) > 100_000:
            return data[:100_000] + "\n... truncated ..."
        return data

    def write_file(self, path: str, content: str) -> str:
        file_path = self._resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"

    def make_directory(self, path: str) -> str:
        directory = self._resolve(path)
        directory.mkdir(parents=True, exist_ok=True)
        return f"Created directory {path}"

    def run_command(self, command: str, cwd: str = ".") -> str:
        self._validate_command(command)
        argv = shlex.split(command)
        if not argv:
            raise ToolError("Command cannot be empty.")

        working_directory = self._resolve(cwd)
        if not working_directory.exists():
            raise ToolError(f"Working directory does not exist: {cwd}")
        if not working_directory.is_dir():
            raise ToolError(f"Working directory is not a directory: {cwd}")

        try:
            result = subprocess.run(
                argv,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"Executable not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Command timed out after {self.command_timeout}s: {command}") from exc

        output = (
            f"cwd={working_directory.relative_to(self.workspace) or Path('.')}\n"
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout[-20_000:]}\n"
            f"stderr:\n{result.stderr[-20_000:]}"
        )
        return output

    def web_search(self, query: str, max_results: int = 5) -> str:
        cleaned = query.strip()
        if not cleaned:
            raise ToolError("Search query cannot be empty.")
        if len(cleaned) > 1000:
            raise ToolError("Search query is too long.")

        max_results = max(1, min(int(max_results), 8))

        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise ToolError(
                "Web search dependency is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            results = list(DDGS().text(cleaned, max_results=max_results))
        except Exception as exc:
            raise ToolError(f"Web search failed: {exc}") from exc

        if not results:
            return f"No public web results found for: {cleaned}"

        lines = [f"WEB SEARCH QUERY: {cleaned}", ""]
        for index, item in enumerate(results, start=1):
            title = str(item.get("title", "Untitled")).strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            body = str(item.get("body") or item.get("snippet") or "").strip()
            lines.append(f"[{index}] {title}")
            if url:
                lines.append(f"URL: {url}")
            if body:
                lines.append(f"SNIPPET: {body}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _validate_command(command: str) -> None:
        normalized = " ".join(command.lower().split())
        blocked_fragments = (
            "sudo ",
            "rm -rf",
            "rm -fr",
            "shutdown",
            "reboot",
            "mkfs",
            "diskutil erase",
            "dd if=",
            "git push",
            "git reset --hard",
            "git clean -fd",
            "curl | sh",
            "curl|sh",
            "wget | sh",
            "wget|sh",
        )
        if any(fragment in normalized for fragment in blocked_fragments):
            raise ToolError("Command blocked by the local safety policy.")

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"Invalid command quoting: {exc}") from exc

        blocked_tokens = {"&&", "||", ";", "|", ">", "<", "`", "$("}
        if any(token in blocked_tokens for token in argv):
            raise ToolError(
                "Shell operators/redirection are disabled. Use run_command with a separate cwd instead of 'cd ... && ...'."
            )
