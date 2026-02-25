"""Agent state definition.

The state is the shared data structure that flows through every node in the graph.
It uses LangGraph's `add_messages` annotation to automatically accumulate messages.
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the agentic graph.

    Attributes:
        messages: Conversation history. Uses `add_messages` reducer so that
                  each node can append messages without overwriting the list.
        iteration_count: Number of agent→tool loop iterations completed.
        pending_tool_call: Stored destructive tool call awaiting user confirmation.
        awaiting_confirmation: Whether the agent is waiting for user confirmation.
        execute_confirmed_tool: Flag set when user approves a destructive action,
                                signals routing to the execution node.
    """

    messages: Annotated[list, add_messages]
    iteration_count: int
    pending_tool_call: Optional[dict]
    awaiting_confirmation: bool
    execute_confirmed_tool: bool
