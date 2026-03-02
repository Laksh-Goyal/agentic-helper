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


# ── Email tool tests ─────────────────────────────────────────────────────────


def test_email_tools_discovered():
    """Both email tools appear in auto-discovered tool list."""
    from tools import get_all_tools

    tool_names = [t.name for t in get_all_tools()]
    assert "draft_email" in tool_names
    assert "send_email" in tool_names


def test_draft_email_saves_file(tmp_path, monkeypatch):
    """draft_email creates a JSON draft file with correct contents."""
    import json

    from agent import config

    monkeypatch.setattr(config, "EMAIL_DRAFTS_DIR", str(tmp_path))

    from tools.email_tool import draft_email

    result = draft_email.invoke({
        "to": "alice@example.com",
        "subject": "Hello",
        "body": "Hi Alice!",
    })
    assert "Draft saved" in result

    drafts = list(tmp_path.glob("*.json"))
    assert len(drafts) == 1

    data = json.loads(drafts[0].read_text())
    assert data["to"] == "alice@example.com"
    assert data["subject"] == "Hello"
    assert data["body"] == "Hi Alice!"


def test_send_email_is_destructive():
    """send_email triggers the destructive-action confirmation check."""
    from agent.guardrails import check_confirmation_needed

    tool_calls = [{
        "name": "send_email",
        "args": {"to": "bob@example.com", "subject": "Test", "body": "Body"},
    }]
    result = check_confirmation_needed(tool_calls)
    assert result is not None
    assert "send_email" in result
    assert "bob@example.com" in result
    assert "Test" in result


def test_send_email_requires_smtp_config(monkeypatch):
    """send_email returns a clear error when SMTP credentials are missing."""
    from agent import config

    monkeypatch.setattr(config, "SMTP_USER", "")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "")

    from tools.email_tool import send_email

    result = send_email.invoke({
        "to": "bob@example.com",
        "subject": "Test",
        "body": "Body",
    })
    assert "SMTP" in result
    assert "not configured" in result


# ── Calendar tool tests ──────────────────────────────────────────────────────


def test_calendar_tools_discovered():
    """Both calendar tools appear in auto-discovered tool list."""
    from tools import get_all_tools

    tool_names = [t.name for t in get_all_tools()]
    assert "list_upcoming_events" in tool_names
    assert "create_calendar_event" in tool_names


def test_create_event_is_destructive():
    """create_calendar_event triggers the destructive-action confirmation check."""
    from agent.guardrails import check_confirmation_needed

    tool_calls = [{
        "name": "create_calendar_event",
        "args": {
            "summary": "Team standup",
            "start_datetime": "2026-03-05T10:00:00+04:00",
            "end_datetime": "2026-03-05T10:30:00+04:00",
        },
    }]
    result = check_confirmation_needed(tool_calls)
    assert result is not None
    assert "create_calendar_event" in result
    assert "Team standup" in result


def test_list_events_not_destructive():
    """list_upcoming_events does NOT trigger confirmation."""
    from agent.guardrails import check_confirmation_needed

    tool_calls = [{
        "name": "list_upcoming_events",
        "args": {"max_results": 5},
    }]
    result = check_confirmation_needed(tool_calls)
    assert result is None


def test_calendar_auth_missing(monkeypatch):
    """list_upcoming_events returns a clear error when credentials are missing."""
    from agent import config

    monkeypatch.setattr(config, "GCAL_CREDENTIALS_FILE", "/tmp/nonexistent_creds.json")
    monkeypatch.setattr(config, "GCAL_TOKEN_FILE", "/tmp/nonexistent_token.json")

    from tools.calendar_tool import list_upcoming_events

    result = list_upcoming_events.invoke({"max_results": 3})
    assert "credentials" in result.lower() or "authenticate" in result.lower()


# ── Task management tests ────────────────────────────────────────────────────


def test_task_tools_discovered():
    """All 5 task tools appear in auto-discovered tool list."""
    from tools import get_all_tools

    tool_names = [t.name for t in get_all_tools()]
    for expected in ("create_project", "list_tasks", "update_task",
                     "add_subtask", "delete_project"):
        assert expected in tool_names, f"'{expected}' not discovered"


def test_create_project_stores_file(tmp_path, monkeypatch):
    """create_project saves a JSON file with correct schema."""
    import json

    from agent import config

    monkeypatch.setattr(config, "TASKS_DIR", str(tmp_path))

    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    project = store.create_project(
        "Test Project",
        "A test project",
        [{"title": "Task A", "estimated_hours": 2}],
    )

    assert project["name"] == "Test Project"
    assert project["slug"] == "test_project"
    assert project["status"] == "active"

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert data["name"] == "Test Project"


def test_create_project_with_subtasks(tmp_path):
    """Subtasks get sequential IDs and correct fields."""
    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    project = store.create_project(
        "Multi-task",
        "desc",
        [
            {"title": "First", "estimated_hours": 3},
            {"title": "Second", "estimated_hours": 5},
            {"title": "Third", "estimated_hours": 2},
        ],
    )
    ids = [s["id"] for s in project["subtasks"]]
    assert ids == [1, 2, 3]
    assert all(s["status"] == "todo" for s in project["subtasks"])
    assert project["subtasks"][1]["estimated_hours"] == 5.0


def test_update_task_status(tmp_path):
    """Status transitions work and record timestamps."""
    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    store.create_project("P", "d", [{"title": "T", "estimated_hours": 1}])

    task = store.update_task("P", 1, "in_progress")
    assert task["status"] == "in_progress"
    assert task["updated_at"] is not None

    task = store.update_task("P", 1, "completed", notes="Done!")
    assert task["status"] == "completed"
    assert task["notes"] == "Done!"


def test_list_tasks_shows_progress(tmp_path):
    """list_tasks output includes progress percentage and timeline."""
    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    store.create_project(
        "Progress Test",
        "desc",
        [
            {"title": "A", "estimated_hours": 4},
            {"title": "B", "estimated_hours": 6},
        ],
    )
    store.update_task("Progress Test", 1, "completed")

    projects = store.list_projects()
    assert len(projects) == 1
    timeline = projects[0]["_timeline"]
    assert timeline["percent_complete"] == 40
    assert timeline["remaining_hours"] == 6.0


def test_add_subtask_increments_id(tmp_path):
    """add_subtask assigns the next sequential ID."""
    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    store.create_project("P", "d", [{"title": "T1", "estimated_hours": 1}])

    new = store.add_subtask("P", "T2", 3.0)
    assert new["id"] == 2
    assert new["title"] == "T2"
    assert new["estimated_hours"] == 3.0


def test_delete_project(tmp_path):
    """delete_project removes the JSON file."""
    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    store.create_project("Del", "d", [{"title": "T", "estimated_hours": 1}])

    assert store.delete_project("Del") is True
    assert store.get_project("Del") is None
    assert list(tmp_path.glob("*.json")) == []


def test_invalid_project_name(tmp_path):
    """Operations on nonexistent projects raise ValueError."""
    from tools.task_store import TaskStore

    store = TaskStore(tasks_dir=str(tmp_path))
    assert store.get_project("nope") is None

    with pytest.raises(ValueError, match="not found"):
        store.update_task("nope", 1, "completed")
