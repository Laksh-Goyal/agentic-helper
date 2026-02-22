"""Smoke tests for the agent graph.

These tests verify the graph compiles correctly and has the expected structure.
They do NOT require API keys — no LLM calls are made.
"""

import pytest


def test_tools_are_discovered():
    """Tool auto-discovery finds the built-in tools."""
    from tools import get_all_tools

    tools = get_all_tools()
    tool_names = [t.name for t in tools]

    assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}: {tool_names}"
    assert "calculator" in tool_names
    assert "get_current_datetime" in tool_names


def test_agent_state_has_messages_key():
    """AgentState schema includes the 'messages' key."""
    from agent.state import AgentState

    assert "messages" in AgentState.__annotations__


def test_graph_compiles():
    """The graph compiles without errors."""
    from agent.graph import graph

    assert graph is not None


def test_graph_has_expected_nodes():
    """The compiled graph has 'agent' and 'tools' nodes."""
    from agent.graph import graph

    # StateGraph exposes node names via .nodes
    node_names = set(graph.get_graph().nodes.keys())
    assert "agent" in node_names, f"'agent' node missing. Found: {node_names}"
    assert "tools" in node_names, f"'tools' node missing. Found: {node_names}"


def test_calculator_tool():
    """Calculator tool evaluates expressions correctly."""
    from tools.calculator import calculator

    result = calculator.invoke({"expression": "2 + 3 * 4"})
    assert result == "14"


def test_datetime_tool():
    """Datetime tool returns a non-empty string."""
    from tools.datetime_tool import get_current_datetime

    result = get_current_datetime.invoke({"timezone_name": "UTC"})
    assert "Current date and time" in result
    assert "UTC" in result
