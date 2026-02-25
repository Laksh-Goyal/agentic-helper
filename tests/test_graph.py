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


def test_agent_state_has_iteration_count():
    """AgentState schema includes the 'iteration_count' key."""
    from agent.state import AgentState

    assert "iteration_count" in AgentState.__annotations__


def test_agent_state_has_confirmation_fields():
    """AgentState schema includes pending_tool_call and awaiting_confirmation."""
    from agent.state import AgentState

    assert "pending_tool_call" in AgentState.__annotations__
    assert "awaiting_confirmation" in AgentState.__annotations__
    assert "execute_confirmed_tool" in AgentState.__annotations__


def test_graph_compiles():
    """The graph compiles without errors."""
    from agent.graph import graph

    assert graph is not None


def test_graph_has_expected_nodes():
    """The compiled graph has all core and sentinel nodes."""
    from agent.graph import graph

    node_names = set(graph.get_graph().nodes.keys())
    for expected in ("agent", "tools", "limit_reached", "needs_confirmation",
                     "handle_confirmation", "execute_confirmed_tool"):
        assert expected in node_names, f"'{expected}' node missing. Found: {node_names}"


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


# ── Guardrail tests ──────────────────────────────────────────────────────────


def test_iteration_limit_routing():
    """_should_continue returns 'limit_reached' when iteration_count >= MAX_ITERATIONS."""
    from unittest.mock import MagicMock

    from agent import config
    from agent.graph import _should_continue

    mock_msg = MagicMock()
    mock_msg.tool_calls = [{"name": "calculator", "args": {}, "id": "1"}]

    state = {"messages": [mock_msg], "iteration_count": config.MAX_ITERATIONS,
             "awaiting_confirmation": False}
    assert _should_continue(state) == "limit_reached"


def test_destructive_confirmation_routing():
    """_should_continue returns 'needs_confirmation' for destructive tool calls."""
    from unittest.mock import MagicMock

    from agent.graph import _should_continue

    mock_msg = MagicMock()
    mock_msg.tool_calls = [{"name": "write_file", "args": {"path": "a.txt", "content": "hi"}, "id": "1"}]

    state = {"messages": [mock_msg], "iteration_count": 0, "awaiting_confirmation": False}
    assert _should_continue(state) == "needs_confirmation"


def test_safe_tool_passes_through():
    """_should_continue returns 'tools' for safe tool calls under the limit."""
    from unittest.mock import MagicMock

    from agent.graph import _should_continue

    mock_msg = MagicMock()
    mock_msg.tool_calls = [{"name": "calculator", "args": {"expression": "1+1"}, "id": "1"}]

    state = {"messages": [mock_msg], "iteration_count": 0, "awaiting_confirmation": False}
    assert _should_continue(state) == "tools"


def test_awaiting_confirmation_routes_to_handler():
    """_should_continue returns 'handle_confirmation' when awaiting_confirmation is True."""
    from unittest.mock import MagicMock

    from agent.graph import _should_continue

    mock_msg = MagicMock()
    mock_msg.tool_calls = []

    state = {"messages": [mock_msg], "iteration_count": 0, "awaiting_confirmation": True}
    assert _should_continue(state) == "handle_confirmation"


def test_confirmation_check_returns_prompt():
    """check_confirmation_needed returns a prompt for destructive tools."""
    from agent.guardrails import check_confirmation_needed

    tool_calls = [{"name": "write_file", "args": {"path": "test.txt", "content": "hello"}}]
    result = check_confirmation_needed(tool_calls)
    assert result is not None
    assert "write_file" in result


def test_confirmation_check_returns_none_for_safe():
    """check_confirmation_needed returns None for safe tools."""
    from agent.guardrails import check_confirmation_needed

    tool_calls = [{"name": "calculator", "args": {"expression": "1+1"}}]
    result = check_confirmation_needed(tool_calls)
    assert result is None


def test_handle_confirmation_approve():
    """handle_confirmation_node returns execute flag on 'yes'."""
    from unittest.mock import MagicMock

    from agent.nodes import handle_confirmation_node

    mock_msg = MagicMock()
    mock_msg.content = "yes"
    state = {"messages": [mock_msg]}

    result = handle_confirmation_node(state)
    assert result.get("execute_confirmed_tool") is True
    assert result.get("awaiting_confirmation") is False


def test_handle_confirmation_deny():
    """handle_confirmation_node cancels and clears state on 'no'."""
    from unittest.mock import MagicMock

    from agent.nodes import handle_confirmation_node

    mock_msg = MagicMock()
    mock_msg.content = "no"
    state = {"messages": [mock_msg]}

    result = handle_confirmation_node(state)
    assert result["pending_tool_call"] is None
    assert result["awaiting_confirmation"] is False
    assert "cancelled" in result["messages"][0].content.lower()


def test_rate_limiter():
    """RateLimiter raises after exceeding max calls."""
    from agent.guardrails import RateLimitExceeded, RateLimiter

    limiter = RateLimiter(max_calls=3, window_seconds=60)
    for _ in range(3):
        limiter.check()

    with pytest.raises(RateLimitExceeded):
        limiter.check()


def test_tool_logger_writes(tmp_path):
    """ToolUsageLogger creates a JSONL file with the expected entry."""
    import json

    from agent.guardrails import ToolUsageLogger

    logger = ToolUsageLogger(log_dir=str(tmp_path))
    logger.log(tool_name="calculator", tool_args={"expression": "1+1"}, result_summary="2")

    log_file = tmp_path / "tool_usage.jsonl"
    assert log_file.exists()

    entry = json.loads(log_file.read_text().strip())
    assert entry["tool"] == "calculator"
    assert entry["args"] == {"expression": "1+1"}
    assert entry["result"] == "2"
