"""Graph node functions.

Each function takes the current AgentState and returns a partial state update.
LangGraph merges the returned dict into the shared state automatically.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from agent import config
from agent.guardrails import RateLimitExceeded, rate_limiter, tool_logger
from memory.store import MemoryStore
from tools import get_all_tools

_memory_store = MemoryStore()


def _build_base_model():
    """Build the bare Gemini LLM *without* tools bound.

    Tool binding now happens per-query inside call_model so we can
    dynamically select only the tools relevant to the user's message.
    """
    return ChatGoogleGenerativeAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY or None,
    )


# Lazily initialised so the module can be imported without side-effects.
_base_model = None


def _get_base_model():
    global _base_model
    if _base_model is None:
        _base_model = _build_base_model()
    return _base_model


def _extract_latest_user_text(messages: list) -> str:
    """Walk backwards to find the most recent human message text."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
        if isinstance(msg, tuple) and msg[0] == "user":
            return msg[1]
    return ""


def _build_system_prompt() -> str:
    """Combine the base system prompt with any stored long-term memory."""
    memory = _memory_store.get_all()
    memory_parts: list[str] = []
    for category in ("personality", "preferences", "key_facts"):
        entries = memory.get(category, {})
        if entries:
            formatted = ", ".join(f"{k}: {v}" for k, v in entries.items())
            memory_parts.append(f"  {category}: {formatted}")

    if memory_parts:
        memory_summary = "Long-term memory:\n" + "\n".join(memory_parts)
        return config.SYSTEM_PROMPT + "\n\n" + memory_summary
    return config.SYSTEM_PROMPT


def call_model(state: dict) -> dict:
    """Invoke the Gemini model with the current conversation history.

    When RAG-based tool retrieval is enabled, only the top-k most
    relevant tools are bound to the model for this invocation.
    Otherwise all tools are bound (original behavior).

    Prepends the system prompt (enriched with long-term memory) as the
    first message if it isn't already there.
    Increments the iteration counter on every call.
    """
    messages = list(state["messages"])

    # Inject (or refresh) system prompt with latest memory
    system_prompt = _build_system_prompt()
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))
    else:
        # Refresh in case memory was updated mid-conversation
        messages[0] = SystemMessage(content=system_prompt)

    base_model = _get_base_model()

    # ── Dynamic tool selection ────────────────────────────────────────────
    if config.TOOL_RETRIEVAL_ENABLED:
        from tools.registry import get_registry
        user_text = _extract_latest_user_text(messages)
        retrieved_tools = get_registry().retrieve(
            user_text, top_k=config.TOOL_RETRIEVAL_TOP_K,
        )
        bound_model = base_model.bind_tools(retrieved_tools) if retrieved_tools else base_model
        available_tool_names = [t.name for t in retrieved_tools]
    else:
        all_tools = get_all_tools()
        bound_model = base_model.bind_tools(all_tools) if all_tools else base_model
        available_tool_names = [t.name for t in all_tools]

    response = bound_model.invoke(messages)

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "available_tools": available_tool_names,
    }


# ── Tool execution with guardrails ───────────────────────────────────────────

_tool_node: ToolNode | None = None


def _get_tool_node() -> ToolNode:
    global _tool_node
    if _tool_node is None:
        _tool_node = ToolNode(get_all_tools())
    return _tool_node


def guarded_tool_node(state: dict) -> dict:
    """Execute tool calls with rate-limiting and logging."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    # ── Rate-limit check ──────────────────────────────────────────────────
    try:
        for _ in tool_calls:
            rate_limiter.check()
    except RateLimitExceeded as exc:
        error_messages = [
            ToolMessage(content=str(exc), tool_call_id=tc["id"])
            for tc in tool_calls
        ]
        return {"messages": error_messages}

    # ── Execute via the standard ToolNode ─────────────────────────────────
    result = _get_tool_node().invoke(state)

    # ── Log each tool call ────────────────────────────────────────────────
    result_messages = result.get("messages", []) if isinstance(result, dict) else []
    for tc, msg in zip(tool_calls, result_messages):
        tool_logger.log(
            tool_name=tc.get("name", "unknown"),
            tool_args=tc.get("args", {}),
            result_summary=getattr(msg, "content", "")[:500],
        )

    return result


# ── Sentinel nodes ────────────────────────────────────────────────────────────


def limit_reached_node(state: dict) -> dict:
    """Emit a message when the iteration limit is hit."""
    return {
        "messages": [
            AIMessage(
                content=(
                    f"⚠️ Iteration limit of {config.MAX_ITERATIONS} reached. "
                    "Stopping to avoid runaway execution. "
                    "You can continue by sending another message."
                )
            )
        ]
    }


# ── Confirmation flow ────────────────────────────────────────────────────────


def needs_confirmation_node(state: dict) -> dict:
    """Store the destructive tool call and ask the user to confirm."""
    from agent.guardrails import check_confirmation_needed

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    prompt = check_confirmation_needed(tool_calls)

    # Store the full last AI message (which contains tool_calls) for later replay
    return {
        "messages": [
            AIMessage(content=prompt or "Confirmation required for the pending action.")
        ],
        "pending_tool_call": {
            "tool_calls": tool_calls,
            "original_message": last_message,
        },
        "awaiting_confirmation": True,
    }


def handle_confirmation_node(state: dict) -> dict:
    """Process the user's yes/no reply to a confirmation prompt."""
    user_reply = state["messages"][-1].content.strip().lower()

    if user_reply in ("yes", "y", "approve"):
        return {
            "execute_confirmed_tool": True,
            "awaiting_confirmation": False,
        }
    else:
        return {
            "messages": [
                AIMessage(content="Action cancelled.")
            ],
            "pending_tool_call": None,
            "awaiting_confirmation": False,
        }


def execute_confirmed_tool_node(state: dict) -> dict:
    """Execute the previously stored destructive tool call after confirmation."""
    pending = state.get("pending_tool_call")
    if not pending:
        return {
            "messages": [AIMessage(content="No pending tool call to execute.")],
            "awaiting_confirmation": False,
            "execute_confirmed_tool": False,
        }

    original_message = pending["original_message"]

    # Re-invoke the tool node with the original AI message at the END
    # of the temporary list so ToolNode identifies the tools to run.
    messages = list(state["messages"])

    # Remove the user's confirmation reply (e.g. "yes").
    # Guard: only pop if it is actually the human confirmation message.
    if (
        messages
        and messages[-1].type == "human"
        and messages[-1].content.strip().lower() in ("yes", "y", "approve")
    ):
        messages.pop()

    # Remove the confirmation prompt AI message that we injected.
    # Guard: only pop a plain AIMessage (no tool_calls) whose content
    # starts with the known confirmation prefix.
    if (
        messages
        and isinstance(messages[-1], AIMessage)
        and not getattr(messages[-1], "tool_calls", None)
        and (
            messages[-1].content.startswith("\u26a0\ufe0f")
            or messages[-1].content.startswith("Confirmation required")
        )
    ):
        messages.pop()

    # Now append original AI tool call
    messages.append(original_message)

    tool_state = {**state, "messages": messages}
    result = _get_tool_node().invoke(tool_state)

    # ── Log the confirmed execution ───────────────────────────────────────
    result_messages = result.get("messages", []) if isinstance(result, dict) else []
    for tc, msg in zip(pending.get("tool_calls", []), result_messages):
        tool_logger.log(
            tool_name=tc.get("name", "unknown"),
            tool_args=tc.get("args", {}),
            result_summary=getattr(msg, "content", "")[:500],
        )

    # ── Prepare update (clearing flags) ───────────────────────────────────
    update = {
        "messages": result_messages,
        "pending_tool_call": None,
        "awaiting_confirmation": False,
        "execute_confirmed_tool": False,
    }
    return update
