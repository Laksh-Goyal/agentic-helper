"""LangGraph graph construction.

Builds a ReAct-style agent graph:

    ┌──────────┐      tool calls?      ┌───────────┐
    │  agent   │ ──────────────────────▶│   tools   │
    │(call_model)│                      │(ToolNode) │
    └──────────┘◀──────────────────────└───────────┘
         │
         │ no tool calls
         ▼
       [END]

The graph loops between agent → tools until the model produces a final
response with no tool calls, at which point it terminates.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.nodes import call_model, make_tool_node
from agent.state import AgentState


def _should_continue(state: dict) -> str:
    """Route to 'tools' if the last message has tool calls, else END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    """Construct and compile the agentic graph."""
    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", make_tool_node())

    # ── Edges ──────────────────────────────────────────────────────────────
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")  # after tool execution, go back to agent

    # ── Compile with checkpointing ─────────────────────────────────────────
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Pre-built graph instance ready to use
graph = build_graph()
