# ⚡ Agentic Helper

A modular, extensible AI agent built with **LangGraph** and **Google Gemini**. Features a ReAct-style reasoning loop, Playwright browser interaction, and a real-time chat UI.

## Architecture

```
┌──────────┐      tool calls?      ┌───────────┐
│  agent   │ ──────────────────────▶│   tools   │
│ (Gemini) │                        │(ToolNode) │
└──────────┘◀──────────────────────└───────────┘
     │
     │ no tool calls
     ▼
   [END]
```

The agent follows a **ReAct loop**: Gemini reasons about the user's request, optionally invokes tools, observes the results, and continues until it has a final answer.

## Project Structure

```
├── agent/              Core agent (state, nodes, graph)
├── tools/              Tool definitions (drop-in extensible)
├── ui/                 Chat UI (FastAPI + WebSocket)
├── scripts/            CLI runner and utilities
└── tests/              Automated tests
```

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
playwright install chromium
```

### 2. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the chat UI

```bash
python ui/server.py
# Open http://localhost:8000
```

### 4. Or use the CLI

```bash
python scripts/run_cli.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to a URL |
| `browser_get_content` | Extract page text |
| `browser_click` | Click an element |
| `browser_type_text` | Type into an input |
| `browser_screenshot` | Capture a screenshot |
| `calculator` | Evaluate math expressions |
| `get_current_datetime` | Get current date/time |

## Adding a New Tool

1. Create a new file in `tools/`, e.g. `tools/my_tool.py`
2. Use the `@tool` decorator from `langchain_core.tools`
3. The tool is automatically discovered — no registration needed

```python
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Description of what this tool does (shown to the LLM)."""
    return f"Result for: {query}"
```

## Configuration

Set via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google AI Studio API key (required) |
| `AGENT_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `AGENT_TEMPERATURE` | `0.0` | LLM sampling temperature |
| `AGENT_MAX_ITERATIONS` | `10` | Max reasoning loop iterations |