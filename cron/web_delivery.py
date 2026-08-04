"""Durable delivery of Cron output into API Server conversations.

The API Server has no long-lived request to write back to after a Cron run.
Its persisted ``SessionDB`` session is the durable delivery target instead.
This module deliberately has no aiohttp dependency so the Cron worker can use
it directly from its background thread.
"""

from __future__ import annotations

import re
from typing import Optional


WEB_DELIVERY_PREFIX = "web:"
AUTOMATION_RESULT_TOOL_NAME = "automation_result"

# HLMate creates ``api-<uuid>`` sessions and Hermes compression keeps this
# conservative identifier shape. Rejecting separators also prevents Cron's
# comma-separated delivery grammar from being reinterpreted as a second target.
_WEB_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


def parse_web_delivery_target(value: object) -> Optional[str]:
    """Return a validated API Server session id from ``web:<session_id>``."""
    if not isinstance(value, str) or not value.startswith(WEB_DELIVERY_PREFIX):
        return None
    session_id = value[len(WEB_DELIVERY_PREFIX):]
    return session_id if _WEB_SESSION_ID_RE.fullmatch(session_id) else None


def web_delivery_target(session_id: str) -> str:
    """Encode one validated API Server session as a Cron delivery value."""
    if not _WEB_SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid Web session ID")
    return f"{WEB_DELIVERY_PREFIX}{session_id}"


def deliver_web_result(*, session_id: str, job_name: str, content: str) -> Optional[str]:
    """Append one Cron result to its API Server conversation.

    A result is stored as a user-role event to preserve strict user/assistant
    alternation when it arrives after an interactive assistant response. The
    dedicated ``tool_name`` marker lets HLMate render it as an automation card;
    the model receives the labelled event as ordinary conversation context.
    """
    if not _WEB_SESSION_ID_RE.fullmatch(session_id):
        return "invalid web conversation target"
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            # A long conversation may have rotated through Hermes compression
            # after the task was created. Always append to its current resume
            # tip so the result appears in the same Web conversation the user
            # now opens, rather than in an obsolete parent row.
            resolved_session_id = db.resolve_resume_session_id(session_id)
            session = db.get_session(resolved_session_id)
            if not session or session.get("source") != "api_server":
                return "web conversation is unavailable or has been deleted"
            db.append_message(
                session_id=resolved_session_id,
                role="user",
                tool_name=AUTOMATION_RESULT_TOOL_NAME,
                content=f"**{job_name}**\n\n{content.strip()}",
            )
        finally:
            db.close()
    except Exception as exc:
        return f"failed to save result to web conversation: {exc}"
    return None
