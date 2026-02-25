"""Persistent JSON-backed memory store.

Reads and writes a single JSON file inside the agent sandbox, structured
by the categories defined in `memory.schema.VALID_CATEGORIES`.
"""

import json
import os
from typing import Any

from agent import config
from memory.schema import VALID_CATEGORIES, UserMemory

_MEMORY_PATH = os.path.join(config.SANDBOX_ROOT, "memory.json")


class MemoryStore:
    """Thread-safe, file-backed key-value store grouped by category."""

    def __init__(self) -> None:
        if not os.path.exists(_MEMORY_PATH):
            self._write({cat: {} for cat in VALID_CATEGORIES})

    # ── Internal I/O ──────────────────────────────────────────────────────

    def _read(self) -> UserMemory:
        with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, category: str, key: str, value: Any) -> None:
        """Set ``data[category][key] = value``.

        Raises:
            ValueError: If *category* is not in VALID_CATEGORIES.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
            )
        data = self._read()
        data[category][key] = value
        self._write(data)

    def get_all(self) -> UserMemory:
        """Return the full memory dict."""
        return self._read()