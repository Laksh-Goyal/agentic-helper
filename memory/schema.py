"""Schema for long-term user memory."""

from typing import Any, Dict, TypedDict

# Canonical set of allowed memory categories.
VALID_CATEGORIES = frozenset({"personality", "preferences", "key_facts"})


class UserMemory(TypedDict):
    personality: Dict[str, Any]
    preferences: Dict[str, Any]
    key_facts: Dict[str, Any]