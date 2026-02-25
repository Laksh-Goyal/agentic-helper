"""Filesystem tools — directory management and sandboxed file I/O.

Gives the agent the ability to explore the filesystem, scaffold
directory structures, and read/write files on behalf of the user.
All paths are resolved inside the sandbox defined in agent.config.
"""

import os

from langchain_core.tools import tool

from agent import config

# ── File I/O constraints ──────────────────────────────────────────────────────
_ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory.

    Returns every file and sub-directory inside the given path, annotated
    with its type (file or dir) and human-readable size.

    Args:
        path: Absolute or relative path to the directory to list.
              Defaults to the current working directory.
    """
    try:
        abs_path = _resolve_path(path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"Error: Path does not exist — {abs_path}"
    if not os.path.isdir(abs_path):
        return f"Error: Path is not a directory — {abs_path}"

    try:
        entries = sorted(os.listdir(abs_path))
    except PermissionError:
        return f"Error: Permission denied — {abs_path}"

    if not entries:
        return f"{abs_path}/ (empty directory)"

    lines: list[str] = [f"Contents of {abs_path}/\n"]
    for entry in entries:
        full = os.path.join(abs_path, entry)
        if os.path.isdir(full):
            lines.append(f"  📁 {entry}/")
        else:
            size = _human_size(os.path.getsize(full))
            lines.append(f"  📄 {entry}  ({size})")

    lines.append(f"\n{len(entries)} item(s)")
    return "\n".join(lines)


@tool
def create_directory(path: str) -> str:
    """Create a directory (and any missing parent directories) inside the sandbox.

    This is safe to call if the directory already exists — it will simply
    report that fact rather than raising an error.

    Args:
        path: Path of the directory to create (resolved inside the sandbox).
    """
    try:
        abs_path = _resolve_path(path)
    except ValueError as e:
        return str(e)

    if os.path.isdir(abs_path):
        return f"Directory already exists — {abs_path}"

    if os.path.exists(abs_path):
        return f"Error: A file (not a directory) already exists at — {abs_path}"

    try:
        os.makedirs(abs_path, exist_ok=True)
        return f"Created directory — {abs_path}"
    except PermissionError:
        return f"Error: Permission denied — cannot create {abs_path}"
    except OSError as e:
        return f"Error creating directory: {e}"


@tool
def append_to_file(path: str, content: str) -> str:
    """Append text content to a file inside the sandbox.

    Creates the file if it does not exist.
    """

    try:
        abs_path = _resolve_path(path)
    except ValueError as e:
        return str(e)

    # Optional: prevent huge writes
    if len(content) > 100_000:
        return "Error: Content too large to append."

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "a", encoding="utf-8") as f:
            f.write(content)

        return f"Appended {len(content)} characters to {abs_path}"

    except PermissionError:
        return f"Error: Permission denied — {abs_path}"
    except OSError as e:
        return f"Error appending to file: {e}"
        

@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file inside the sandbox (overwrites if it exists).

    Only the following file types are allowed: .txt, .md, .json, .csv.
    Maximum file size is 5 MB.

    Args:
        path: File path (resolved inside the sandbox).
        content: The text content to write.
    """
    try:
        abs_path = _resolve_path(path)
    except ValueError as e:
        return str(e)

    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return (
            f"Error: File type '{ext}' is not allowed. "
            f"Permitted extensions: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    if len(content.encode("utf-8")) > _MAX_FILE_SIZE:
        return f"Error: Content exceeds the {_MAX_FILE_SIZE // (1024 * 1024)} MB limit."

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {abs_path}"
    except PermissionError:
        return f"Error: Permission denied — {abs_path}"
    except OSError as e:
        return f"Error writing file: {e}"


@tool
def read_file(path: str) -> str:
    """Read the text content of a file inside the sandbox.

    Only the following file types are allowed: .txt, .md, .json, .csv.
    Files larger than 5 MB are rejected.

    Args:
        path: File path (resolved inside the sandbox).
    """
    try:
        abs_path = _resolve_path(path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"Error: File does not exist — {abs_path}"
    if not os.path.isfile(abs_path):
        return f"Error: Path is not a file — {abs_path}"

    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return (
            f"Error: File type '{ext}' is not allowed. "
            f"Permitted extensions: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    size = os.path.getsize(abs_path)
    if size > _MAX_FILE_SIZE:
        return f"Error: File is {_human_size(size)}, exceeds the {_MAX_FILE_SIZE // (1024 * 1024)} MB limit."

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except PermissionError:
        return f"Error: Permission denied — {abs_path}"
    except UnicodeDecodeError:
        return f"Error: File is not valid UTF-8 text — {abs_path}"
    except OSError as e:
        return f"Error reading file: {e}"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _human_size(num_bytes: int) -> str:
    """Convert a byte count to a compact human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"


def _resolve_path(path: str) -> str:
    """Resolve a path inside the sandbox. Reject escapes."""
    abs_path = os.path.abspath(os.path.join(config.SANDBOX_ROOT, path))

    if os.path.commonpath([abs_path, config.SANDBOX_ROOT]) != config.SANDBOX_ROOT:
        raise ValueError("Access denied: Path escapes sandbox")

    return abs_path