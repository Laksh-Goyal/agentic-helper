"""Date and time tool — returns the current date and time."""

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time.

    Use this when the user asks about the current time, date, day of the week,
    or any time-related question.

    Args:
        timezone_name: Timezone name (default: UTC). Common values:
                       'UTC', 'US/Eastern', 'US/Pacific', 'Europe/London', 'Asia/Dubai'
    """
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(timezone_name)
    except (ImportError, KeyError):
        tz = timezone.utc
        timezone_name = "UTC (fallback)"

    now = datetime.now(tz)
    return (
        f"Current date and time ({timezone_name}): "
        f"{now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}"
    )
