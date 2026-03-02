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

DESTRUCTIVE_TOOLS: list[str] = [
    "write_file", "append_to_file", "create_directory",
    "browser_click", "browser_type_text",
    "send_email",
    "create_calendar_event",
]

TOOL_LOG_DIR: str = os.path.join(SANDBOX_ROOT, ".tool_logs")
os.makedirs(TOOL_LOG_DIR, exist_ok=True)

# ── Tool retrieval (RAG) ─────────────────────────────────────────────────────
TOOL_RETRIEVAL_ENABLED: bool = os.getenv("TOOL_RETRIEVAL_ENABLED", "true").lower() == "true"
TOOL_RETRIEVAL_TOP_K: int = int(os.getenv("TOOL_RETRIEVAL_TOP_K", "3"))
TOOL_EMBEDDING_MODEL: str = os.getenv("TOOL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOOL_INDEX_DIR: str = os.path.join(SANDBOX_ROOT, ".tool_index")
os.makedirs(TOOL_INDEX_DIR, exist_ok=True)

# ── Email / SMTP ──────────────────────────────────────────────────────────────
SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM: str = os.getenv("EMAIL_FROM", "")
EMAIL_DRAFTS_DIR: str = os.path.join(SANDBOX_ROOT, ".email_drafts")
os.makedirs(EMAIL_DRAFTS_DIR, exist_ok=True)

# ── Google Calendar ───────────────────────────────────────────────────────────
_GCAL_DIR = os.path.join(SANDBOX_ROOT, ".gcal_credentials")
os.makedirs(_GCAL_DIR, exist_ok=True)
GCAL_CREDENTIALS_FILE: str = os.getenv(
    "GOOGLE_CALENDAR_CREDENTIALS_FILE",
    os.path.join(_GCAL_DIR, "credentials.json"),
)
GCAL_TOKEN_FILE: str = os.getenv(
    "GOOGLE_CALENDAR_TOKEN_FILE",
    os.path.join(_GCAL_DIR, "token.json"),
)

# ── Task Management ──────────────────────────────────────────────────────────
TASKS_DIR: str = os.path.join(SANDBOX_ROOT, ".tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

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
