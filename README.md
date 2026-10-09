# PTA Autonomous Developer Agent

A local-first autonomous software-development agent that can plan a task, inspect a workspace, create and edit files, run build/test commands, recover from failures, and verify completion.

The agent can use:

- **OpenAI Responses API** for high-quality planning, coding, debugging, and review.
- **Ollama** for local/private model execution and lower-cost tasks.
- A **controlled tool layer** for file operations and terminal execution.

## What v1 can do

1. Accept a natural-language development task.
2. Ask the configured model for a concise implementation plan.
3. Inspect the target workspace.
4. Repeatedly choose and execute tools:
   - list files
   - read files
   - write files
   - create directories
   - run safe local commands
5. Feed command output and errors back to the model.
6. Continue until the model marks the task complete or the step limit is reached.
7. Print a final summary.

## Safety model

The agent is intentionally sandboxed to a configured workspace directory. File paths may not escape that directory. High-risk commands such as `sudo`, destructive recursive deletion, disk formatting, shutdown/reboot, and `git push` are blocked by the local executor.

This is a development agent, not a guarantee of perfect software. The goal is to make completion measurable by builds, tests, runtime checks, and explicit acceptance criteria.

## Requirements

- Python 3.11+
- Optional: Ollama running locally
- Optional: OpenAI API key

> A ChatGPT subscription and an OpenAI API key are separate products. To use the OpenAI provider from this program, configure an API key.

## Setup

```bash
git clone https://github.com/prabhat143/agent.git
cd agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and choose your provider.

### OpenAI

```env
AGENT_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.3-codex
```

### Ollama

Start Ollama and pull a coding model, for example:

```bash
ollama pull qwen3-coder
ollama serve
```

Then configure:

```env
AGENT_PROVIDER=ollama
OLLAMA_MODEL=qwen3-coder
OLLAMA_BASE_URL=http://localhost:11434
```

You can see local Ollama models with:

```bash
ollama list
```

## Run

The default workspace is `./workspace`.

```bash
python main.py
```

Or pass a task directly:

```bash
python main.py "Create a FastAPI health-check service with tests"
```

Use a different workspace:

```bash
python main.py --workspace ../my-project "Add validation and unit tests to the user API"
```

## Agent loop

```text
TASK
  ↓
PLAN
  ↓
INSPECT WORKSPACE
  ↓
CHOOSE TOOL
  ↓
EXECUTE
  ↓
OBSERVE RESULT
  ↓
SUCCESS? ── no ──> DEBUG / FIX ──┐
  │                               │
 yes <────────────────────────────┘
  ↓
VERIFY
  ↓
DONE
```

## Current v1 limits

- No autonomous remote Git push.
- No browser-control tool yet.
- No persistent vector memory yet.
- Model decisions use a small JSON action protocol rather than unrestricted shell access.

These are deliberate v1 choices. Browser testing, GitHub PR delivery, richer memory, multi-agent delegation, and approval policies can be added next.
