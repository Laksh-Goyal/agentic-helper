"""Agent safety guardrails — logging, rate-limiting, and confirmation checks.

All guardrail logic lives here to keep nodes.py and graph.py focused on
their primary responsibilities.
"""

import json
import os
import time
from collections import deque
from typing import Optional

from agent import config


# ── Tool-usage logger ─────────────────────────────────────────────────────────


class ToolUsageLogger:
    """Append-only JSON-lines logger for every tool invocation."""

    def __init__(self, log_dir: str = config.TOOL_LOG_DIR):
        self._log_path = os.path.join(log_dir, "tool_usage.jsonl")

    def log(
        self,
        tool_name: str,
        tool_args: dict,
        result_summary: str,
    ) -> None:
        """Write a single log entry."""
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": tool_name,
            "args": tool_args,
            "result": result_summary[:500],  # keep logs compact
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# Shared singleton
tool_logger = ToolUsageLogger()


# ── Rate limiter ──────────────────────────────────────────────────────────────


class RateLimitExceeded(Exception):
    """Raised when the agent exceeds the tool-call rate limit."""


class RateLimiter:
    """Sliding-window rate limiter for tool calls."""

    def __init__(
        self,
        max_calls: int = config.RATE_LIMIT_CALLS,
        window_seconds: int = config.RATE_LIMIT_WINDOW,
    ):
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    def check(self) -> None:
        """Record a call and raise if the limit is exceeded."""
        now = time.time()
        # Evict timestamps outside the window
        while self._timestamps and self._timestamps[0] <= now - self._window:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._max_calls:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self._max_calls} tool calls "
                f"within {self._window}s. Please wait before retrying."
            )
        self._timestamps.append(now)


# Shared singleton
rate_limiter = RateLimiter()


# ── Destructive-action confirmation ──────────────────────────────────────────


def check_confirmation_needed(tool_calls: list[dict]) -> Optional[str]:
    """Return a confirmation prompt if any tool call is destructive.

    Resolves each target path inside the sandbox and reports whether the
    file or directory already exists (overwrite) or is new.

    Args:
        tool_calls: List of tool-call dicts (each has 'name' and 'args').

    Returns:
        A human-readable prompt string, or None if no confirmation is needed.
    """
    destructive = [
        tc for tc in tool_calls
        if tc.get("name") in config.DESTRUCTIVE_TOOLS
    ]
    if not destructive:
        return None

    descriptions = []
    for tc in destructive:
        name = tc["name"]
        args = tc.get("args", {})
        path_arg = args.get("path", "")

        # Resolve path inside the sandbox and check existence
        exists_note = ""
        if path_arg:
            try:
                abs_path = os.path.abspath(
                    os.path.join(config.SANDBOX_ROOT, path_arg)
                )
                if name == "create_directory":
                    if os.path.isdir(abs_path):
                        exists_note = " ⚠ directory already exists"
                    elif os.path.exists(abs_path):
                        exists_note = " ⚠ a file already exists at this path"
                else:  # write_file, append_to_file
                    if os.path.isfile(abs_path):
                        size = os.path.getsize(abs_path)
                        exists_note = f" ⚠ file already exists ({size} bytes — will be {'overwritten' if name == 'write_file' else 'appended to'})"
                    elif os.path.exists(abs_path):
                        exists_note = " ⚠ path exists but is not a regular file"
                    else:
                        exists_note = " (new file)"
            except (ValueError, OSError):
                exists_note = " (could not verify path)"

        args_summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
        descriptions.append(f"  • {name}({args_summary}){exists_note}")

    return (
        "⚠️  The following destructive action(s) require your confirmation:\n"
        + "\n".join(descriptions)
        + "\n\nPlease confirm to proceed, or reply to cancel."
    )
