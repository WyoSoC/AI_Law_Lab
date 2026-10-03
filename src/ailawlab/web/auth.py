"""Signing in through Keycloak, and the gate in front of every page.

The sign-in is OpenID Connect's authorization-code flow with PKCE, written out here rather
than taken from a library because one detail is unusual: this server cannot reach its own
public address. Browsers are sent to Keycloak's public URL (which is also the tokens'
issuer), while this server exchanges the code and fetches signing keys over loopback
(settings.oidc_internal_url). The ID token is verified -- signature, issuer, audience,
expiry, nonce -- before anything in it is believed.

The session is a signed cookie (Starlette's SessionMiddleware) holding the user's id and
the ID token (for Keycloak's sign-out). Every request re-reads the user from the database,
so disabling someone takes effect on their next click, not when their cookie expires.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from joserfc import jwt
from joserfc.errors import JoseError
from joserfc.jwk import KeySet

from .. import accounts
from ..config import settings

log = logging.getLogger(__name__)
router = APIRouter()

# Paths (below the mount point) anyone may open without signing in.
PUBLIC_EXACT = {"/", "/auth/login", "/auth/callback", "/auth/logout", "/auth/signed-out"}
PUBLIC_PREFIXES = ("/static/",)
# Signed in but not (yet) allowed in: these still work, everything else explains why not.
LIMBO_EXACT = {"/auth/pending", "/auth/disabled"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# A person playing a role in a role-play may take their turns with a view-only account: the
# seat endpoints check that the seat is theirs.
SEAT_PATH = re.compile(r"^/api/runs/[0-9a-f-]{36}/seat/[a-z]+$")

# Stand-in user when settings.auth_required is off (local development only).
DEV_USER = {"id": None, "name": "Local developer", "email": "", "role": "admin",
            "status": "active", "identity_provider": "", "dev": True}


# ---------------------------------------------------------------- pure helpers


def safe_next(target: str | None, prefix: str) -> str:
    """Where to go after signing in: a path on this site only, never another host."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return f"{prefix}/dashboard"
    if prefix and not (target == prefix or target.startswith(prefix + "/")):
        return f"{prefix}/dashboard"
    return target


def origin_allowed(origin: str | None, public_url: str) -> bool:
    """Whether a state-changing request's Origin header is this site's.

    Browsers send Origin on every cross-site POST, so a mismatch means another site is
    trying to act with the visitor's cookie. No Origin (curl, some old browsers) is let
    through: the SameSite=Lax cookie already keeps cross-site form posts unauthenticated.
    """
    if not origin or origin == "null":
        return origin is None
    want = urlsplit(public_url)
    got = urlsplit(origin)
    return (got.scheme, got.netloc) == (want.scheme, want.netloc)


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def may(user: dict | None, action: str) -> bool:
    """Role check. Actions: 'read', 'write' (create, run, curate), 'admin'."""
    if not user or user.get("status") != "active":
        return False
    role = user.get("role")
    return {"read": role in ("viewer", "researcher", "admin"),
            "write": role in ("researcher", "admin"),
            "admin": role == "admin"}.get(action, False)


# ---------------------------------------------------------------- OIDC


_jwks: dict[str, Any] = {"keys": None, "at": 0.0}


async def _keys(force: bool = False) -> KeySet:
    if force or _jwks["keys"] is None or time.time() - _jwks["at"] > 3600:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{settings.oidc_internal_url}/protocol/openid-connect/certs")
            r.raise_for_status()
        _jwks.update(keys=KeySet.import_key_set(r.json()), at=time.time())
    return _jwks["keys"]


async def verify_id_token(token: str, nonce: str) -> dict[str, Any]:
    try:
        keys = await _keys()
        try:
            decoded = jwt.decode(token, keys, algorithms=["RS256", "ES256", "PS256"])
        except JoseError:
            # Keycloak may have rotated its signing key since the last fetch.
            decoded = jwt.decode(token, await _keys(force=True), algorithms=["RS256", "ES256", "PS256"])
        jwt.JWTClaimsRegistry(
            iss={"essential": True, "value": settings.oidc_issuer},
            aud={"essential": True, "value": settings.oidc_client_id},
            sub={"essential": True}, exp={"essential": True},
            nonce={"essential": True, "value": nonce},
            leeway=60,
        ).validate(decoded.claims)
    except JoseError as e:
        raise HTTPException(400, f"The sign-in could not be verified ({type(e).__name__}). "
                                 "Try signing in again.") from e
    return dict(decoded.claims)


def _callback_url() -> str:
    return f"{settings.public_url.rstrip('/')}/auth/callback"


@router.get("/auth/login")
async def login(request: Request, next: str | None = None, provider: str | None = None):
    """Send the browser to Keycloak. `provider` (uwyo, google, microsoft) skips Keycloak's
    own page and goes straight to that sign-in."""
    prefix = request.scope.get("root_path", "")
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    verifier, challenge = pkce_pair()
    request.session["oidc"] = {"state": state, "nonce": nonce, "verifier": verifier,
                               "next": safe_next(next, prefix), "at": int(time.time())}
    params = {"client_id": settings.oidc_client_id, "response_type": "code",
              "scope": "openid email profile", "redirect_uri": _callback_url(),
              "state": state, "nonce": nonce, "code_challenge": challenge,
              "code_challenge_method": "S256"}
    if provider in ("uwyo", "google", "microsoft"):
        params["kc_idp_hint"] = provider
    return RedirectResponse(f"{settings.oidc_issuer}/protocol/openid-connect/auth?{urlencode(params)}",
                            status_code=303)


@router.get("/auth/callback")
async def callback(request: Request, code: str | None = None, state: str | None = None,
                   error: str | None = None):
    prefix = request.scope.get("root_path", "")
    pending = request.session.pop("oidc", None)
    if error:
        # The person cancelled, or Keycloak refused; either way start over cleanly.
        return RedirectResponse(f"{prefix}/?signin=cancelled", status_code=303)
    if not pending or not code or not state or not secrets.compare_digest(state, pending["state"]) \
            or time.time() - pending["at"] > 900:
        raise HTTPException(400, "This sign-in link has expired or was already used. Sign in again.")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{settings.oidc_internal_url}/protocol/openid-connect/token",
                         data={"grant_type": "authorization_code", "code": code,
                               "redirect_uri": _callback_url(), "code_verifier": pending["verifier"]},
                         auth=(settings.oidc_client_id, settings.oidc_client_secret))
    if r.status_code != 200:
        log.warning("token exchange failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(400, "The sign-in could not be completed. Sign in again.")
    tokens = r.json()
    claims = await verify_id_token(tokens["id_token"], pending["nonce"])
    user = await accounts.sign_in(claims)
    request.session.clear()
    request.session.update({"uid": str(user["id"]), "idt": tokens["id_token"],
                            "since": int(time.time())})
    if user["status"] == "pending":
        return RedirectResponse(f"{prefix}/auth/pending", status_code=303)
    return RedirectResponse(pending["next"], status_code=303)


@router.get("/auth/logout")
async def logout(request: Request):
    """End the app session and the Keycloak session, then come back to the About page."""
    prefix = request.scope.get("root_path", "")
    id_token = request.session.get("idt")
    request.session.clear()
    back = f"{settings.public_url.rstrip('/')}/auth/signed-out"
    params = {"client_id": settings.oidc_client_id, "post_logout_redirect_uri": back}
    if id_token:
        params["id_token_hint"] = id_token
    if not settings.auth_required:
        return RedirectResponse(f"{prefix}/", status_code=303)
    return RedirectResponse(f"{settings.oidc_issuer}/protocol/openid-connect/logout?{urlencode(params)}",
                            status_code=303)


@router.get("/auth/signed-out")
async def signed_out(request: Request):
    return RedirectResponse(f"{request.scope.get('root_path', '')}/?signin=out", status_code=303)


# ---------------------------------------------------------------- the gate


class AuthGate:
    """ASGI middleware: resolve the signed-in user and keep everyone else out.

    Written as plain ASGI rather than BaseHTTPMiddleware so the live run stream (SSE) and
    its disconnect detection pass through untouched. Must sit inside SessionMiddleware.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        root = scope.get("root_path", "")
        path = scope["path"]
        if root and path.startswith(root):
            path = path[len(root):] or "/"
        method = scope["method"]
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        state = scope.setdefault("state", {})

        user = await self._user(scope)
        state["user"] = user
        if user and may(user, "admin") and not user.get("dev") and method == "GET":
            state["pending_accounts"] = await accounts.pending_count()

        if method not in SAFE_METHODS and not origin_allowed(headers.get("origin"), settings.public_url) \
                and settings.auth_required:
            return await self._deny(scope, receive, send, 403,
                                    "This request came from another site and was refused.")

        public = path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIXES)
        if public:
            return await self.app(scope, receive, send)
        wants_html = method == "GET" and not path.startswith("/api/") and \
            "text/html" in headers.get("accept", "text/html")
        if user is None:
            if wants_html:
                target = f"{root}{path}" + (f"?{scope['query_string'].decode()}" if scope.get("query_string") else "")
                return await self._redirect(scope, receive, send,
                                            f"{root}/auth/login?{urlencode({'next': target})}")
            return await self._deny(scope, receive, send, 401, "Sign in to use AI Law Lab.")
        if user["status"] != "active":
            limbo = "/auth/pending" if user["status"] == "pending" else "/auth/disabled"
            if path in LIMBO_EXACT:
                return await self.app(scope, receive, send)
            if wants_html:
                return await self._redirect(scope, receive, send, f"{root}{limbo}")
            return await self._deny(scope, receive, send, 403,
                                    "Your account is waiting for approval." if user["status"] == "pending"
                                    else "Your account has been disabled.")
        if method not in SAFE_METHODS and not may(user, "write") and path != "/auth/logout" \
                and not SEAT_PATH.match(path):
            return await self._deny(scope, receive, send, 403,
                                    "Your account can view the lab but not change it. Ask an "
                                    "administrator for researcher access.")
        return await self.app(scope, receive, send)

    async def _user(self, scope) -> dict | None:
        if not settings.auth_required:
            return DEV_USER
        session = scope.get("session") or {}
        uid = session.get("uid")
        if not uid:
            return None
        try:
            return await accounts.get_user(uid)
        except Exception:  # noqa: BLE001 - a malformed or stale session is just signed out
            log.warning("could not load session user %r", uid)
            return None

    @staticmethod
    async def _redirect(scope, receive, send, location: str):
        return await RedirectResponse(location, status_code=303)(scope, receive, send)

    @staticmethod
    async def _deny(scope, receive, send, status: int, message: str):
        return await JSONResponse({"detail": message}, status_code=status)(scope, receive, send)


def current_user(request: Request) -> dict | None:
    return getattr(request.state, "user", None)


def require(request: Request, action: str) -> dict:
    """The signed-in user, if their role allows `action` ('write' or 'admin'); else 403."""
    user = current_user(request)
    if not may(user, action):
        raise HTTPException(403, "Only an administrator can do that." if action == "admin"
                            else "Your account can view the lab but not change it.")
    return user


def user_id(request: Request) -> Any:
    """The signed-in user's id for attribution, or None (development mode)."""
    user = current_user(request)
    return user.get("id") if user else None


def no_store(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    return response
