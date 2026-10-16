"""Email about access requests: admins hear of a new one, the person hears the decision.

Sent through the outgoing mail server in settings (the one Keycloak uses). With none
configured, or if sending fails, nothing is sent and the reason is logged: a missed email
never blocks a request or a decision, since the Users page shows every request anyway.
"""
from __future__ import annotations

import asyncio
import logging
import smtplib
from email.message import EmailMessage

from .config import settings

log = logging.getLogger(__name__)


def mail_ready() -> bool:
    return bool(settings.smtp_host and settings.smtp_from)


def _send(to: list[str], subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = f"AI Law Lab <{settings.smtp_from}>"
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as s:
        if settings.smtp_starttls:
            s.starttls()
        if settings.smtp_user:
            s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)


async def send(to: list[str], subject: str, body: str) -> bool:
    to = [a for a in to if a]
    if not to or not mail_ready():
        return False
    try:
        await asyncio.to_thread(_send, to, subject, body)
        return True
    except (OSError, ValueError, smtplib.SMTPException) as e:
        log.warning("could not send %r to %s: %s", subject, to, e)
        return False


def _users_url() -> str:
    return f"{settings.public_url.rstrip('/')}/admin/users"


async def request_received(user: dict, admins: list[str]) -> bool:
    via = {"uwyo": "UW sign-on", "google": "Google", "microsoft": "Microsoft"}.get(
        user.get("identity_provider") or "", "an AI Law Lab account")
    body = (f"{user['name']} <{user['email']}> has asked for {user['requested_role']} access "
            f"to the AI Law Lab.\n\n"
            f"Organization: {user['affiliation']}\nSigned in with: {via}\n\n"
            f"What they would like to use it for:\n{user['request_reason']}\n\n"
            f"Approve or decline it on the Users page: {_users_url()}\n")
    return await send(admins, f"Access request from {user['name']}", body)


async def request_decided(user: dict, approved: bool) -> bool:
    site = settings.public_url.rstrip("/")
    if approved:
        body = (f"Hello {user['name'] or ''},\n\nYour request to use the AI Law Lab has been "
                f"approved, with {user['role']} access. Sign in at {site}/ with the same "
                f"account you used before.\n")
        subject = "Your AI Law Lab access is approved"
    else:
        note = (user.get("decision_note") or "").strip()
        body = (f"Hello {user['name'] or ''},\n\nYour request to use the AI Law Lab was not "
                f"approved." + (f"\n\nNote from the administrator:\n{note}" if note else "") + "\n")
        subject = "Your AI Law Lab access request"
    return await send([user["email"]], subject, body)
