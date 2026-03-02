"""Persistent JSON-backed task/project store.

Each project is stored as a separate JSON file in the tasks directory.
Provides CRUD operations, ID management, and timeline computation.
Follows the same pattern as ``memory.store.MemoryStore``.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from agent import config

_VALID_STATUSES = {"todo", "in_progress", "completed"}
_PRODUCTIVE_HOURS_PER_DAY = 6.0


def _slugify(text: str, max_len: int = 60) -> str:
    """Turn *text* into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    """File-backed project and task manager.

    Each project is a JSON file at ``<TASKS_DIR>/<slug>.json``.
    """

    def __init__(self, tasks_dir: str = config.TASKS_DIR) -> None:
        self._dir = tasks_dir
        os.makedirs(self._dir, exist_ok=True)

    # ── Internal I/O ──────────────────────────────────────────────────────

    def _project_path(self, slug: str) -> str:
        return os.path.join(self._dir, f"{slug}.json")

    def _read(self, slug: str) -> dict:
        path = self._project_path(slug)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, slug: str, data: dict) -> None:
        path = self._project_path(slug)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _find_slug(self, name: str) -> Optional[str]:
        """Resolve a project name to its slug.

        Tries exact slug match first, then scans files for a name match.
        """
        slug = _slugify(name)
        if os.path.exists(self._project_path(slug)):
            return slug

        # Fallback: scan all projects for a case-insensitive name match
        for filename in os.listdir(self._dir):
            if not filename.endswith(".json"):
                continue
            try:
                data = self._read(filename[:-5])
                if data.get("name", "").lower() == name.lower():
                    return data.get("slug", filename[:-5])
            except (json.JSONDecodeError, OSError):
                continue
        return None

    # ── Public API ────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str,
        subtasks: list[dict],
    ) -> dict:
        """Create a new project with the given subtasks.

        Args:
            name: Human-readable project name.
            description: Project description.
            subtasks: List of dicts with 'title' and optional 'estimated_hours'.

        Returns:
            The full project dict.

        Raises:
            ValueError: If a project with this name already exists.
        """
        slug = _slugify(name)
        if os.path.exists(self._project_path(slug)):
            raise ValueError(f"Project '{name}' already exists.")

        now = _now_iso()
        project = {
            "name": name,
            "slug": slug,
            "description": description,
            "status": "active",
            "created_at": now,
            "subtasks": [],
        }

        for i, st in enumerate(subtasks, 1):
            project["subtasks"].append({
                "id": i,
                "title": st.get("title", f"Task {i}"),
                "status": "todo",
                "estimated_hours": float(st.get("estimated_hours", 1.0)),
                "notes": "",
                "created_at": now,
                "updated_at": None,
            })

        self._write(slug, project)
        return project

    def get_project(self, name: str) -> Optional[dict]:
        """Load a project by name or slug. Returns None if not found."""
        slug = self._find_slug(name)
        if slug is None:
            return None
        try:
            return self._read(slug)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def list_projects(self) -> list[dict]:
        """Return all projects with computed progress info."""
        projects = []
        for filename in sorted(os.listdir(self._dir)):
            if not filename.endswith(".json"):
                continue
            try:
                data = self._read(filename[:-5])
                data["_timeline"] = self.compute_timeline(data)
                projects.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return projects

    def update_task(
        self,
        project_name: str,
        task_id: int,
        status: str,
        notes: str = "",
    ) -> dict:
        """Update a subtask's status and optional notes.

        Args:
            project_name: Name or slug of the project.
            task_id: Numeric ID of the subtask.
            status: New status — 'todo', 'in_progress', or 'completed'.
            notes: Optional note to attach.

        Returns:
            The updated subtask dict.

        Raises:
            ValueError: If project/task not found or status is invalid.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {', '.join(sorted(_VALID_STATUSES))}"
            )

        slug = self._find_slug(project_name)
        if slug is None:
            raise ValueError(f"Project '{project_name}' not found.")

        project = self._read(slug)

        task = None
        for st in project["subtasks"]:
            if st["id"] == task_id:
                task = st
                break

        if task is None:
            raise ValueError(
                f"Task #{task_id} not found in project '{project_name}'."
            )

        task["status"] = status
        task["updated_at"] = _now_iso()
        if notes:
            task["notes"] = notes

        # Auto-derive project status
        all_done = all(s["status"] == "completed" for s in project["subtasks"])
        project["status"] = "completed" if all_done else "active"

        self._write(slug, project)
        return task

    def add_subtask(
        self,
        project_name: str,
        title: str,
        estimated_hours: float = 1.0,
    ) -> dict:
        """Append a new subtask to an existing project.

        Returns:
            The new subtask dict.

        Raises:
            ValueError: If the project is not found.
        """
        slug = self._find_slug(project_name)
        if slug is None:
            raise ValueError(f"Project '{project_name}' not found.")

        project = self._read(slug)
        max_id = max((s["id"] for s in project["subtasks"]), default=0)

        subtask = {
            "id": max_id + 1,
            "title": title,
            "status": "todo",
            "estimated_hours": float(estimated_hours),
            "notes": "",
            "created_at": _now_iso(),
            "updated_at": None,
        }
        project["subtasks"].append(subtask)

        # Un-complete project if it was done
        project["status"] = "active"
        self._write(slug, project)
        return subtask

    def delete_project(self, project_name: str) -> bool:
        """Delete a project file.

        Returns:
            True if deleted, False if not found.
        """
        slug = self._find_slug(project_name)
        if slug is None:
            return False
        path = self._project_path(slug)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ── Timeline computation ──────────────────────────────────────────────

    @staticmethod
    def compute_timeline(project: dict) -> dict:
        """Compute progress and timeline estimates for a project.

        Returns a dict with:
            total_hours, completed_hours, remaining_hours,
            percent_complete, estimated_days_remaining
        """
        subtasks = project.get("subtasks", [])
        if not subtasks:
            return {
                "total_hours": 0,
                "completed_hours": 0,
                "remaining_hours": 0,
                "percent_complete": 0,
                "estimated_days_remaining": 0,
            }

        total = sum(s["estimated_hours"] for s in subtasks)
        completed = sum(
            s["estimated_hours"]
            for s in subtasks
            if s["status"] == "completed"
        )
        remaining = total - completed
        pct = round((completed / total) * 100) if total > 0 else 0
        days = math.ceil(remaining / _PRODUCTIVE_HOURS_PER_DAY) if remaining > 0 else 0

        return {
            "total_hours": total,
            "completed_hours": completed,
            "remaining_hours": remaining,
            "percent_complete": pct,
            "estimated_days_remaining": days,
        }
