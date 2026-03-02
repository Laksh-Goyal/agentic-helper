"""Task management tools — project decomposition and progress tracking.

Provides five tools for managing projects and subtasks:

* ``create_project``  – break down a new project into subtasks
* ``list_tasks``      – view progress across all projects or one project
* ``update_task``     – change a subtask's status
* ``add_subtask``     – add a new subtask to an existing project
* ``delete_project``  – remove a project entirely

All data is persisted as JSON files in ``workspace/.tasks/``.
"""

import json

from langchain_core.tools import tool

from tools.task_store import TaskStore

_store = TaskStore()


def _progress_bar(pct: int, width: int = 20) -> str:
    """Render a text-based progress bar."""
    filled = round(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct}%"


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def create_project(name: str, description: str, subtasks_json: str) -> str:
    """Create a new project broken down into actionable subtasks.

    Before calling this tool, analyze the user's project and decompose
    it into 3–15 concrete, actionable subtasks. Each subtask should be
    specific enough to be completed in a single work session. Include
    realistic hour estimates for each subtask.

    Args:
        name: Project name (e.g. 'Website Redesign').
        description: Brief description of the project's goal.
        subtasks_json: JSON array string of subtasks. Each object must
                       have 'title' (str) and 'estimated_hours' (number).
                       Example: '[{"title": "Design mockups", "estimated_hours": 4}]'
    """
    try:
        subtasks = json.loads(subtasks_json)
    except json.JSONDecodeError as e:
        return f"❌ Invalid subtasks_json: {e}"

    if not isinstance(subtasks, list) or not subtasks:
        return "❌ subtasks_json must be a non-empty JSON array."

    for i, st in enumerate(subtasks):
        if not isinstance(st, dict) or "title" not in st:
            return f"❌ Subtask #{i + 1} is missing a 'title' field."

    try:
        project = _store.create_project(name, description, subtasks)
    except ValueError as e:
        return f"❌ {e}"

    timeline = _store.compute_timeline(project)

    lines = [
        f"✅ Project created: **{project['name']}**",
        f"   {project['description']}",
        f"",
        f"📋 {len(project['subtasks'])} subtask(s):",
    ]
    for st in project["subtasks"]:
        lines.append(f"   {st['id']}. {st['title']}  ({st['estimated_hours']}h)")

    lines.extend([
        f"",
        f"⏱️  Total estimated: {timeline['total_hours']}h "
        f"(~{timeline['estimated_days_remaining']} day(s) at 6h/day)",
    ])
    return "\n".join(lines)


@tool
def list_tasks(project_name: str = "") -> str:
    """List tasks and progress for your projects.

    Use this when the user asks about their current tasks, project status,
    what they've completed, what's remaining, or project timelines.

    Args:
        project_name: Optional — name of a specific project to show.
                      If omitted, shows a summary of all projects.
    """
    if project_name:
        # ── Detailed view for one project ─────────────────────────────
        project = _store.get_project(project_name)
        if project is None:
            return f"❌ Project '{project_name}' not found."

        timeline = _store.compute_timeline(project)
        lines = [
            f"📂 **{project['name']}**",
            f"   {project['description']}",
            f"   {_progress_bar(timeline['percent_complete'])}",
            f"",
        ]

        # Group by status
        groups = {
            "🔵 In Progress": [],
            "🟡 To Do": [],
            "✅ Completed": [],
        }
        for st in project["subtasks"]:
            if st["status"] == "in_progress":
                groups["🔵 In Progress"].append(st)
            elif st["status"] == "todo":
                groups["🟡 To Do"].append(st)
            else:
                groups["✅ Completed"].append(st)

        for label, tasks in groups.items():
            if tasks:
                lines.append(f"   {label}:")
                for t in tasks:
                    notes_tag = f'  — "{t["notes"]}"' if t.get("notes") else ""
                    lines.append(
                        f"     {t['id']}. {t['title']}  "
                        f"({t['estimated_hours']}h){notes_tag}"
                    )
                lines.append("")

        lines.extend([
            f"⏱️  {timeline['completed_hours']}h done / "
            f"{timeline['remaining_hours']}h remaining / "
            f"{timeline['total_hours']}h total",
            f"📅 ~{timeline['estimated_days_remaining']} day(s) to completion "
            f"(at 6h/day)",
        ])
        return "\n".join(lines)

    # ── Overview of all projects ──────────────────────────────────────
    projects = _store.list_projects()
    if not projects:
        return "📋 No projects yet. Use create_project to get started."

    lines = [f"📋 **{len(projects)} Project(s)**\n"]
    for p in projects:
        t = p["_timeline"]
        status_icon = "✅" if p["status"] == "completed" else "📂"
        lines.append(
            f"  {status_icon} **{p['name']}** — "
            f"{_progress_bar(t['percent_complete'])}"
        )
        done_count = sum(1 for s in p["subtasks"] if s["status"] == "completed")
        total_count = len(p["subtasks"])
        lines.append(
            f"      {done_count}/{total_count} tasks · "
            f"{t['remaining_hours']}h remaining · "
            f"~{t['estimated_days_remaining']} day(s) left"
        )
        lines.append("")

    return "\n".join(lines)


@tool
def update_task(
    project_name: str,
    task_id: int,
    status: str,
    notes: str = "",
) -> str:
    """Update the status of a subtask in a project.

    Use this when the user starts working on a task, completes a task,
    or wants to add notes to a task.

    Args:
        project_name: Name of the project containing the task.
        task_id: Numeric ID of the subtask to update.
        status: New status — must be 'todo', 'in_progress', or 'completed'.
        notes: Optional note to attach (e.g. 'Blocked on design review').
    """
    try:
        task = _store.update_task(project_name, task_id, status, notes)
    except ValueError as e:
        return f"❌ {e}"

    status_icons = {"todo": "🟡", "in_progress": "🔵", "completed": "✅"}
    icon = status_icons.get(task["status"], "")

    result = (
        f"{icon} Task #{task['id']} updated: **{task['title']}**\n"
        f"   Status: {task['status']}"
    )
    if task.get("notes"):
        result += f"\n   Notes: {task['notes']}"
    return result


@tool
def add_subtask(
    project_name: str,
    title: str,
    estimated_hours: float = 1.0,
) -> str:
    """Add a new subtask to an existing project.

    Use this when the user identifies additional work needed for a project
    that wasn't in the original plan.

    Args:
        project_name: Name of the project to add the subtask to.
        title: Title of the new subtask.
        estimated_hours: Estimated hours to complete (default: 1.0).
    """
    try:
        subtask = _store.add_subtask(project_name, title, estimated_hours)
    except ValueError as e:
        return f"❌ {e}"

    return (
        f"✅ Subtask added to project:\n"
        f"   #{subtask['id']}. {subtask['title']}  ({subtask['estimated_hours']}h)"
    )


@tool
def delete_project(project_name: str) -> str:
    """Delete a project and all its tasks.

    Use this when a project is no longer relevant or was created by mistake.

    Args:
        project_name: Name of the project to delete.
    """
    if _store.delete_project(project_name):
        return f"✅ Project '{project_name}' deleted."
    return f"❌ Project '{project_name}' not found."
