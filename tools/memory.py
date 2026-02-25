"""LangChain tools for reading and updating long-term memory."""

from langchain_core.tools import tool

from memory.schema import VALID_CATEGORIES
from memory.store import MemoryStore

_store = MemoryStore()


@tool
def update_memory(category: str, key: str, value: str) -> str:
    """Update a single entry in long-term memory.

    Use this when the user shares personal information, preferences, or
    facts worth remembering across conversations.

    Args:
        category: Must be one of 'personality', 'preferences', or 'key_facts'.
        key:      Short label for the memory entry (e.g. 'tone', 'location').
        value:    The value to store.
    """
    if category not in VALID_CATEGORIES:
        return (
            f"Error: Invalid category '{category}'. "
            f"Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
        )
    try:
        _store.update(category, key, value)
        return f"Memory updated: {category}.{key} = {value}"
    except Exception as e:
        return f"Error updating memory: {e}"


@tool
def read_memory() -> str:
    """Return all stored long-term memory as a formatted string.

    Use this when you need to recall facts, preferences, or personality
    traits the user has previously shared.
    """
    data = _store.get_all()
    if not any(data.values()):
        return "No long-term memories stored yet."

    lines: list[str] = []
    for category, entries in data.items():
        if entries:
            lines.append(f"{category}:")
            for k, v in entries.items():
                lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "No long-term memories stored yet."
