"""LangGraph graph construction.

Builds a ReAct-style agent graph with safety guardrails:

    ┌──────────┐      tool calls?      ┌───────────┐
    │  agent   │ ─────────────────────▶│   tools   │
    │(call_model)│                      │(guarded)  │
    └──────────┘◀─────────────────────└───────────┘
         │
         ├─ iteration limit ──▶ [limit_reached] ──▶ END
         ├─ destructive ──────▶ [needs_confirmation] ──▶ END (awaits user reply)
         │
         └─ no tool calls ──▶ END

On re-entry with awaiting_confirmation=True:

    [agent] ──▶ [handle_confirmation]
                   ├─ approved ──▶ [execute_confirmed_tool] ──▶ [agent]
                   └─ denied ───▶ END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent import config
from agent.guardrails import check_confirmation_needed
from agent.nodes import (
    call_model,
    execute_confirmed_tool_node,
    guarded_tool_node,
    handle_confirmation_node,
    limit_reached_node,
    needs_confirmation_node,
)
from agent.state import AgentState


def _should_continue(state: dict) -> str:
    """Route based on tool calls, iteration limits, and destructive actions."""
    last_message = state["messages"][-1]

    # 1. If we're awaiting confirmation, route to confirmation handler
    if state.get("awaiting_confirmation", False):
        return "handle_confirmation"

    # 2. No tool calls → done
    if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return END

    # 3. Iteration limit
    if state.get("iteration_count", 0) >= config.MAX_ITERATIONS:
        return "limit_reached"

    # 4. Destructive-action confirmation
    if check_confirmation_needed(last_message.tool_calls):
        return "needs_confirmation"

    return "tools"


def _after_confirmation(state: dict) -> str:
    """Route after handle_confirmation_node based on user's decision."""
    if state.get("execute_confirmed_tool"):
        return "execute_confirmed_tool"
    return END


def _route_start(state: dict) -> str:
    """Route to confirmation handler if we're waiting for user input, else to agent."""
    if state.get("awaiting_confirmation"):
        return "handle_confirmation"
    return "agent"


def build_graph() -> StateGraph:
    """Construct and compile the agentic graph."""
    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", guarded_tool_node)
    workflow.add_node("limit_reached", limit_reached_node)
    workflow.add_node("needs_confirmation", needs_confirmation_node)
    workflow.add_node("handle_confirmation", handle_confirmation_node)
    workflow.add_node("execute_confirmed_tool", execute_confirmed_tool_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    workflow.set_conditional_entry_point(
        _route_start,
        {
            "agent": "agent",
            "handle_confirmation": "handle_confirmation",
        },
    )

    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "tools": "tools",
            "limit_reached": "limit_reached",
            "needs_confirmation": "needs_confirmation",
            "handle_confirmation": "handle_confirmation",
            END: END,
        },
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("limit_reached", END)
    workflow.add_edge("needs_confirmation", END)  # awaits user reply

    workflow.add_conditional_edges(
        "handle_confirmation",
        _after_confirmation,
        {
            "execute_confirmed_tool": "execute_confirmed_tool",
            END: END,
        },
    )

    workflow.add_edge("execute_confirmed_tool", "agent")

    # ── Compile with checkpointing ─────────────────────────────────────────
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Pre-built graph instance ready to use
graph = build_graph()
