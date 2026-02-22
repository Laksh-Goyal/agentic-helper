"""Graph node functions.

Each function takes the current AgentState and returns a partial state update.
LangGraph merges the returned dict into the shared state automatically.
"""

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from agent import config
from tools import get_all_tools


def _build_model():
    """Build and return the Gemini LLM with tools bound."""
    tools = get_all_tools()
    llm = ChatGoogleGenerativeAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY or None,
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


# Lazily initialised so the module can be imported without side-effects.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = _build_model()
    return _model


def call_model(state: dict) -> dict:
    """Invoke the Gemini model with the current conversation history.

    Prepends the system prompt as the first message if it isn't already there.
    """
    messages = list(state["messages"])

    # Inject system prompt if missing
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=config.SYSTEM_PROMPT))

    model = _get_model()
    response = model.invoke(messages)
    return {"messages": [response]}


def make_tool_node() -> ToolNode:
    """Create a ToolNode that can execute any registered tool."""
    tools = get_all_tools()
    return ToolNode(tools)
