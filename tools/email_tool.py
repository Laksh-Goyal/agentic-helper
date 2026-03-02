"""Email tool — draft and send emails.

Provides two tools:

* ``draft_email``  – compose and save a draft to disk (safe, no confirmation).
* ``send_email``   – send an email via SMTP (destructive, requires user
  confirmation through the existing guardrail flow).

SMTP settings are read from environment variables / ``agent.config``.
"""

import json
import os
import re
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText

from langchain_core.tools import tool

from agent import config


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn *text* into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len]


def _save_draft(to: str, subject: str, body: str) -> str:
    """Persist a draft as a JSON file and return the path."""
    os.makedirs(config.EMAIL_DRAFTS_DIR, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{ts}_{_slugify(subject)}.json"
    path = os.path.join(config.EMAIL_DRAFTS_DIR, filename)

    draft = {
        "to": to,
        "subject": subject,
        "body": body,
        "created_at": ts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2)

    return path


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """Compose an email draft and save it to disk for review.

    Use this tool when the user wants to compose, prepare, or draft an email
    without sending it immediately. The draft is saved as a JSON file that
    can be reviewed or edited later.

    Args:
        to: Recipient email address (e.g. 'alice@example.com').
        subject: Email subject line.
        body: Full body text of the email.
    """
    try:
        path = _save_draft(to, subject, body)
        return (
            f"✅ Draft saved successfully.\n"
            f"  To: {to}\n"
            f"  Subject: {subject}\n"
            f"  Path: {path}"
        )
    except Exception as e:
        return f"❌ Failed to save draft: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient via SMTP.

    Use this tool when the user explicitly wants to send an email.
    This action requires user confirmation before execution because
    sending email is irreversible.

    The email will be sent using the configured SMTP server. Make sure
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and EMAIL_FROM
    environment variables are set.

    Args:
        to: Recipient email address (e.g. 'alice@example.com').
        subject: Email subject line.
        body: Full body text of the email.
    """
    # Validate SMTP configuration
    if not config.SMTP_USER or not config.SMTP_PASSWORD:
        return (
            "❌ SMTP credentials not configured. "
            "Please set SMTP_USER and SMTP_PASSWORD environment variables."
        )
    if not config.EMAIL_FROM:
        return (
            "❌ Sender address not configured. "
            "Please set the EMAIL_FROM environment variable."
        )

    # Also save a copy as a draft for records
    try:
        draft_path = _save_draft(to, subject, body)
    except Exception:
        draft_path = None

    # Build the message
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_FROM
    msg["To"] = to

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)

        result = (
            f"✅ Email sent successfully.\n"
            f"  To: {to}\n"
            f"  Subject: {subject}"
        )
        if draft_path:
            result += f"\n  Draft copy: {draft_path}"
        return result

    except smtplib.SMTPAuthenticationError:
        return "❌ SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD."
    except smtplib.SMTPException as e:
        return f"❌ Failed to send email: {e}"
    except Exception as e:
        return f"❌ Unexpected error sending email: {e}"
