"""Calculator tool — evaluate mathematical expressions safely.

Uses Python's `ast` module to parse and evaluate expressions, restricting
to numeric literals and basic arithmetic operators.  No `eval()` is used.
"""

import ast
import operator

from langchain_core.tools import tool

# Map AST node types to the operators they represent.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node containing only numbers and arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # Guard against unreasonably large exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return _OPERATORS[type(node.op)](left, right)

    raise ValueError(
        f"Unsupported expression element: {ast.dump(node)}. "
        "Only numbers and arithmetic operators (+, -, *, /, //, %, **) are allowed."
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool when you need to perform calculations. Supports standard
    arithmetic operations: +, -, *, /, **, //, %, and parentheses.

    Args:
        expression: A mathematical expression to evaluate, e.g. "2 + 3 * 4"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as e:
        return f"Error evaluating expression: {e}"
