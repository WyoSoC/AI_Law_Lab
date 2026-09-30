"""People who use the lab: who they are, whether they are approved, and in what role.

Keycloak proves who someone is (with UW sign-on, Google, Microsoft, or an account of its
own); this module decides what that person may do. On a first sign-in:

* an address listed in settings.admin_emails, if the provider vouches for it, becomes an
  active admin -- how the first admin exists without anyone to approve them;
* a verified address in an auto-approve domain (uwyo.edu) becomes an active researcher,
  unless it came through Microsoft, whose work-account addresses are set by whoever runs
  each organization's tenant and so prove nothing (Keycloak is told not to trust them);
* everyone else is pending until an admin approves them.

Roles: a viewer reads everything and exports; a researcher also creates experiments, runs
them and curates libraries; an admin also manages people and deletes things for good.
"""
from __future__ import annotations

from typing import Any

from .config import settings
from .db import fetch_all, fetch_one, jsonb

ROLES = ("admin", "researcher", "viewer")
STATUSES = ("pending", "active", "disabled")

# Providers whose "email_verified" is not proof that the person owns the address.
UNTRUSTED_EMAIL_PROVIDERS = frozenset({"microsoft"})


def _listed(value: str) -> set[str]:
    return {v.strip().lower() for v in value.split(",") if v.strip()}


def email_is_proven(email: str, verified: bool, provider: str) -> bool:
    return bool(email) and verified and provider not in UNTRUSTED_EMAIL_PROVIDERS


def initial_access(email: str, verified: bool, provider: str,
                   admin_emails: str, auto_domains: str) -> tuple[str, str]:
    """(status, role) for someone signing in for the first time. Pure."""
    email = email.strip().lower()
    if email_is_proven(email, verified, provider):
        if email in _listed(admin_emails):
            return "active", "admin"
        domain = email.rsplit("@", 1)[-1]
        if domain in _listed(auto_domains):
            return "active", "researcher"
    return "pending", "researcher"


def display_name(user: dict[str, Any] | None) -> str:
    if not user:
        return ""
    return user.get("name") or user.get("email") or "Unnamed user"


# ---------------------------------------------------------------- storage


async def get_user(user_id: str) -> dict | None:
    return await fetch_one("SELECT * FROM users WHERE id=%s", (user_id,))


async def sign_in(claims: dict[str, Any]) -> dict:
    """Record a sign-in from verified ID-token claims; create the user on first sign-in.

    Name and email follow the identity provider on every sign-in. A listed admin address,
    once proven, is made an active admin even if the account already existed, so an admin
    locked out by a mistaken role change can recover by signing in again.
    """
    email = str(claims.get("email") or "").strip().lower()
    verified = bool(claims.get("email_verified"))
    provider = str(claims.get("identity_provider") or "")
    name = str(claims.get("name") or " ".join(
        p for p in (claims.get("given_name"), claims.get("family_name")) if p) or "").strip()
    existing = await fetch_one("SELECT * FROM users WHERE subject=%s", (claims["sub"],))
    if existing is None:
        status, role = initial_access(email, verified, provider, settings.admin_emails,
                                      settings.auto_approve_domains)
        user = await fetch_one(
            "INSERT INTO users (subject, email, email_verified, name, identity_provider, status, "
            "role, last_login_at, approved_at) VALUES (%s,%s,%s,%s,%s,%s,%s, now(), "
            "CASE WHEN %s = 'active' THEN now() END) RETURNING *",
            (claims["sub"], email, verified, name, provider, status, role, status))
        await audit(user["id"], "account.created", email,
                    status=status, role=role, provider=provider or "keycloak")
        return user
    user = await fetch_one(
        "UPDATE users SET email=%s, email_verified=%s, name=%s, identity_provider=%s, "
        "last_login_at=now() WHERE id=%s RETURNING *",
        (email, verified, name or existing["name"], provider, existing["id"]))
    if (email_is_proven(email, verified, provider) and email in _listed(settings.admin_emails)
            and (user["role"] != "admin" or user["status"] != "active")):
        user = await fetch_one("UPDATE users SET role='admin', status='active' WHERE id=%s "
                               "RETURNING *", (user["id"],))
        await audit(user["id"], "account.admin_restored", email)
    return user


async def list_users() -> list[dict]:
    return await fetch_all(
        "SELECT u.*, a.name AS approved_by_name, a.email AS approved_by_email FROM users u "
        "LEFT JOIN users a ON a.id = u.approved_by "
        "ORDER BY (u.status = 'pending') DESC, u.created_at DESC")


async def pending_count() -> int:
    row = await fetch_one("SELECT COUNT(*) AS n FROM users WHERE status='pending'")
    return row["n"] if row else 0


class AccountError(ValueError):
    """A change to someone's access that is not allowed, in words an admin can act on."""


async def set_access(actor: dict, user_id: str, *, status: str | None = None,
                     role: str | None = None) -> dict:
    """Approve, disable, re-enable, or change the role of an account."""
    if status is not None and status not in STATUSES:
        raise AccountError(f"Unknown status {status!r}.")
    if role is not None and role not in ROLES:
        raise AccountError(f"Unknown role {role!r}.")
    target = await get_user(user_id)
    if target is None:
        raise AccountError("No such account.")
    losing_admin = target["role"] == "admin" and target["status"] == "active" and (
        (role is not None and role != "admin") or (status is not None and status != "active"))
    if losing_admin:
        others = await fetch_one("SELECT COUNT(*) AS n FROM users WHERE role='admin' "
                                 "AND status='active' AND id <> %s", (user_id,))
        if not others or not others["n"]:
            raise AccountError("This is the only active admin. Make someone else an admin first.")
    approving = status == "active" and target["status"] == "pending"
    user = await fetch_one(
        "UPDATE users SET status=COALESCE(%s, status), role=COALESCE(%s, role), "
        "approved_by = CASE WHEN %s THEN %s ELSE approved_by END, "
        "approved_at = CASE WHEN %s THEN now() ELSE approved_at END "
        "WHERE id=%s RETURNING *",
        (status, role, approving, actor["id"], approving, user_id))
    await audit(actor["id"], "account.access", target["email"] or target["subject"],
                status=user["status"], role=user["role"],
                was={"status": target["status"], "role": target["role"]})
    return user


async def audit(user_id: Any, action: str, target: str = "", **detail: Any) -> None:
    await fetch_one("INSERT INTO audit_log (user_id, action, target, detail) VALUES (%s,%s,%s,%s) "
                    "RETURNING id", (user_id, action, target, jsonb(detail)))


async def recent_audit(limit: int = 50) -> list[dict]:
    return await fetch_all(
        "SELECT l.*, u.name AS user_name, u.email AS user_email FROM audit_log l "
        "LEFT JOIN users u ON u.id = l.user_id ORDER BY l.at DESC LIMIT %s", (limit,))
