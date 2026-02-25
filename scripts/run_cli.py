"""Interactive CLI for testing the agent without the web UI.

Usage:
    python scripts/run_cli.py
"""

import atexit
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import graph  # noqa: E402
from tools.browser import close_browser  # noqa: E402

# Ensure Playwright resources are freed on exit
atexit.register(close_browser)


def main():
    """Run an interactive chat loop in the terminal."""
    print("=" * 60)
    print("  ⚡ Agentic Helper — CLI Mode")
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
        try:
            for event in graph.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="values",
            ):
                messages = event.get("messages", [])
                if messages:
                    last = messages[-1]

                    # Show tool calls
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        for tc in last.tool_calls:
                            name = tc.get("name", "unknown")
                            args = tc.get("args", {})
                            print(f"  \033[1;33m🔧 Tool: {name}\033[0m")
                            for k, v in args.items():
                                print(f"     {k}: {v}")

                    # Show tool results
                    elif last.type == "tool":
                        result = last.content[:300]
                        print(f"  \033[1;32m✅ {last.name}:\033[0m {result}")

                    # Show final response
                    elif last.type == "ai" and last.content and not (
                        hasattr(last, "tool_calls") and last.tool_calls
                    ):
                        print(f"\033[1;35mAgent:\033[0m {last.content}")

        except Exception as e:
            print(f"\033[1;31mError:\033[0m {e}")

        print()


if __name__ == "__main__":
    main()
