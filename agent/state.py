"""Agent state definition.

The state is the shared data structure that flows through every node in the graph.
It uses LangGraph's `add_messages` annotation to automatically accumulate messages.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the agentic graph.

    Attributes:
        messages: Conversation history. Uses `add_messages` reducer so that
                  each node can append messages without overwriting the list.
    """

    messages: Annotated[list, add_messages]
