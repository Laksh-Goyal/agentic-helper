"""Interactive CLI for testing the agent without the web UI.

Usage:
    python scripts/run_cli.py
"""

import atexit
import json
import os
import sys
import time
from datetime import datetime

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import graph  # noqa: E402
from tools.browser import close_browser  # noqa: E402

# Ensure Playwright resources are freed on exit
atexit.register(close_browser)


def serialize_message(m):
    """Helper to convert LangChain messages to a dict for logging."""
    if isinstance(m, tuple):
        return {"type": m[0], "content": m[1]}
    msg_dict = {"type": getattr(m, "type", "unknown"), "content": getattr(m, "content", repr(m))}
    if hasattr(m, "tool_calls") and m.tool_calls:
        msg_dict["tool_calls"] = m.tool_calls
    if getattr(m, "name", None):
        msg_dict["name"] = m.name
    return msg_dict


def main():
    """Run an interactive chat loop in the terminal."""
    # Ensure logs directory exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    session_log_path = os.path.join(log_dir, f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    print("=" * 60)
    print("  ⚡ Agentic Helper — CLI Mode")
    print(f"  📝 Logging session to: {session_log_path}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    thread_id = "cli-session"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print()

        # Stream the agent response
        # stream_mode="updates" yields {node_name: state_update} so we can
        # display which graph node is executing at each step.
        
        # Log the user request first
        with open(session_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"user_input": user_input, "timestamp": time.time()}) + "\\n")
            
        try:
            step_start_time = time.time()
            for event in graph.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="updates",
            ):
                event_time = time.time()
                latency = round(event_time - step_start_time, 2)
                step_start_time = event_time  # Reset for next event

                for node_name, state_update in event.items():
                    # ── Display Available Tools ───────────────────────────
                    if "available_tools" in state_update:
                        tools = state_update["available_tools"]
                        if tools:
                            print(f"     \\033[1;36m🔍 Retrieved tools bound:\\033[0m {', '.join(tools)}")

                    # ── Log Event to File ─────────────────────────────────
                    messages_to_log = [serialize_message(m) for m in state_update.get("messages", [])]
                    log_entry = {
                        "timestamp": event_time,
                        "node": node_name,
                        "latency_seconds": latency,
                        "available_tools": state_update.get("available_tools", []),
                        "messages": messages_to_log
                    }
                    with open(session_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\\n")

                    # ── Graph node header ─────────────────────────────────
                    node_icons = {
                        "agent": "🤖",
                        "tools": "🛠 ",
                        "needs_confirmation": "⚠️ ",
                        "handle_confirmation": "🔐",
                        "execute_confirmed_tool": "✔️ ",
                        "limit_reached": "🛑",
                    }
                    icon = node_icons.get(node_name, "➤")
                    print(f"  \033[1;34m{icon}  [{node_name}]\033[0m")

                    messages = state_update.get("messages", [])
                    if not messages:
                        continue
                    last = messages[-1]

                    # Show tool calls made by the agent
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        for tc in last.tool_calls:
                            name = tc.get("name", "unknown")
                            args = tc.get("args", {})
                            print(f"     \033[1;33m🔧 Calling: {name}\033[0m")
                            for k, v in args.items():
                                print(f"        {k}: {v}")

                    # Show tool results
                    elif last.type == "tool":
                        result = last.content[:300]
                        print(f"     \033[1;32m✅ Result ({last.name}):\033[0m {result}")

                    # Show final AI response
                    elif last.type == "ai" and last.content and not (
                        hasattr(last, "tool_calls") and last.tool_calls
                    ):
                        print(f"\n\033[1;35mAgent:\033[0m {last.content}")

        except Exception as e:
            print(f"\033[1;31mError:\033[0m {e}")

        print()



if __name__ == "__main__":
    main()
