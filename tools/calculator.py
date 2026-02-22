"""Calculator tool — evaluate mathematical expressions safely."""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool when you need to perform calculations. Supports standard
    arithmetic operations: +, -, *, /, **, //, %, and parentheses.

    Args:
        expression: A mathematical expression to evaluate, e.g. "2 + 3 * 4"
    """
    # Restrict to safe characters only
    allowed_chars = set("0123456789+-*/.() %e")
    if not all(c in allowed_chars for c in expression.replace(" ", "")):
        return f"Error: Expression contains invalid characters. Only numbers and basic operators (+, -, *, /, **, //, %) are allowed."

    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
