"""Tool auto-registration.

Automatically discovers and collects all `@tool`-decorated functions
from sibling modules. To add a new tool, simply create a new .py file
in this directory with functions decorated with `@langchain_core.tools.tool`.
"""

import importlib
import pkgutil
from typing import Callable

from langchain_core.tools import BaseTool


def get_all_tools() -> list[BaseTool]:
    """Scan the tools package and return all registered tool instances."""
    tool_list: list[BaseTool] = []

    # Iterate over every module in this package
    package_path = __path__  # type: ignore[name-defined]
    for importer, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if module_name.startswith("_"):
            continue
        module = importlib.import_module(f"tools.{module_name}")

        # Collect anything that is a BaseTool or list of BaseTools
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BaseTool):
                tool_list.append(attr)
            elif isinstance(attr, list) and all(isinstance(t, BaseTool) for t in attr):
                tool_list.extend(attr)

    return tool_list
