"""Centralized agent configuration.

Reads from environment variables with sensible defaults so that the agent
works out of the box while remaining fully customizable.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ── LLM Settings ──────────────────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("AGENT_MODEL", "gemini-2.0-flash")
TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
SANDBOX_ROOT = os.path.abspath("./workspace")
os.makedirs(SANDBOX_ROOT, exist_ok=True)

# ── Safety & rate-limiting ────────────────────────────────────────────────────
RATE_LIMIT_CALLS: int = int(os.getenv("AGENT_RATE_LIMIT_CALLS", "30"))
RATE_LIMIT_WINDOW: int = int(os.getenv("AGENT_RATE_LIMIT_WINDOW", "60"))  # seconds

DESTRUCTIVE_TOOLS: list[str] = ["write_file", "append_to_file", "create_directory"]

TOOL_LOG_DIR: str = os.path.join(SANDBOX_ROOT, ".tool_logs")
os.makedirs(TOOL_LOG_DIR, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    (
        "You are a helpful AI assistant with access to a set of tools. "
        "Use the tools when appropriate to answer the user's questions. "
        "Think step by step and explain your reasoning. "
        "When you use a tool, explain why you chose it and what you expect to learn."
    ),
)
