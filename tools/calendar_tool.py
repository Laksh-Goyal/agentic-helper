"""Google Calendar tool — list and create calendar events.

Provides two tools:

* ``list_upcoming_events`` – retrieve upcoming events from the user's
  primary Google Calendar (safe, no confirmation).
* ``create_calendar_event`` – create a new event (destructive, requires
  user confirmation through the existing guardrail flow).

Authentication uses OAuth2 via a ``credentials.json`` file obtained from
the Google Cloud Console. On first use a browser-based consent screen
opens; subsequent calls reuse the persisted ``token.json``.
"""

import os
from datetime import datetime, timezone

from langchain_core.tools import tool

from agent import config

# Google API imports (deferred inside helpers so the module can be
# imported even when the libraries aren't installed yet).

_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_calendar_service():
    """Build and return an authorized Google Calendar API service object.

    Raises a clear error if the credentials file is missing.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    token_path = config.GCAL_TOKEN_FILE
    creds_path = config.GCAL_CREDENTIALS_FILE

    # 1. Load existing token
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, _SCOPES)

    # 2. Refresh or run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Google Calendar credentials file not found at "
                    f"'{creds_path}'. Download an OAuth client-ID JSON "
                    f"from the Google Cloud Console and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, _SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for next time
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def list_upcoming_events(max_results: int = 5) -> str:
    """List upcoming events from your Google Calendar.

    Use this tool when the user asks about their schedule, upcoming
    meetings, appointments, or what's on their calendar.

    Args:
        max_results: Maximum number of events to return (default: 5, max: 20).
    """
    max_results = min(max(1, max_results), 20)

    try:
        service = _get_calendar_service()
    except FileNotFoundError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ Failed to authenticate with Google Calendar: {e}"

    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now_iso,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = result.get("items", [])
        if not events:
            return "📅 No upcoming events found on your calendar."

        lines = [f"📅 Next {len(events)} upcoming event(s):\n"]
        for i, event in enumerate(events, 1):
            summary = event.get("summary", "(No title)")
            start = event["start"].get("dateTime", event["start"].get("date", ""))
            end = event["end"].get("dateTime", event["end"].get("date", ""))
            location = event.get("location", "")

            # Format datetimes for readability
            try:
                start_dt = datetime.fromisoformat(start)
                start_fmt = start_dt.strftime("%a, %b %d %Y at %I:%M %p")
            except (ValueError, TypeError):
                start_fmt = start  # all-day event or unparseable

            try:
                end_dt = datetime.fromisoformat(end)
                end_fmt = end_dt.strftime("%I:%M %p")
            except (ValueError, TypeError):
                end_fmt = end

            entry = f"  {i}. {summary}\n     📆 {start_fmt} – {end_fmt}"
            if location:
                entry += f"\n     📍 {location}"
            lines.append(entry)

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error fetching calendar events: {e}"


@tool
def create_calendar_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
) -> str:
    """Create a new event on your Google Calendar.

    Use this tool when the user wants to schedule a meeting, appointment,
    reminder, or any calendar event. This action requires user confirmation
    before execution because it modifies the calendar.

    Args:
        summary: Title of the event (e.g. 'Team standup').
        start_datetime: Start time in ISO 8601 format with timezone
                        (e.g. '2026-03-05T10:00:00+04:00').
        end_datetime: End time in ISO 8601 format with timezone
                      (e.g. '2026-03-05T11:00:00+04:00').
        description: Optional detailed description of the event.
        location: Optional location or meeting link.
    """
    # Validate datetime format
    for label, dt_str in [("start", start_datetime), ("end", end_datetime)]:
        try:
            datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            return (
                f"❌ Invalid {label} datetime '{dt_str}'. "
                f"Use ISO 8601 format, e.g. '2026-03-05T10:00:00+04:00'."
            )

    try:
        service = _get_calendar_service()
    except FileNotFoundError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ Failed to authenticate with Google Calendar: {e}"

    event_body = {
        "summary": summary,
        "start": {"dateTime": start_datetime},
        "end": {"dateTime": end_datetime},
    }
    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location

    try:
        created = (
            service.events()
            .insert(calendarId="primary", body=event_body)
            .execute()
        )

        link = created.get("htmlLink", "")
        start_dt = datetime.fromisoformat(start_datetime)
        end_dt = datetime.fromisoformat(end_datetime)

        result = (
            f"✅ Event created successfully.\n"
            f"  Title: {summary}\n"
            f"  When: {start_dt.strftime('%a, %b %d %Y at %I:%M %p')} – "
            f"{end_dt.strftime('%I:%M %p')}"
        )
        if location:
            result += f"\n  Where: {location}"
        if link:
            result += f"\n  Link: {link}"
        return result

    except Exception as e:
        return f"❌ Error creating calendar event: {e}"
