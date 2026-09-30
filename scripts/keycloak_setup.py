"""Configure the AI Law Lab realm in Keycloak. Safe to run again after changing .env.

    .venv/bin/python scripts/keycloak_setup.py [--admin-email you@uwyo.edu]

Talks to Keycloak's admin API over loopback (http://127.0.0.1:8180/sso), as a console
admin: KC_ADMIN_USERNAME from .env (asking for the password at the terminal, which is not
stored), or else the temporary bootstrap admin while it still exists.
It creates or updates:

* the `ailawlab` realm: sign-in by email, brute-force lockout, a password policy, and
  self-registration only when an outgoing mail server is configured (registration needs
  email verification, and password resets need email);
* the `ai-law-lab` client the web app signs people in through, and its secret, which is
  written to .env as AILAWLAB_OIDC_CLIENT_SECRET if not already there;
* a token claim naming the identity provider someone signed in with;
* each identity provider whose credentials are in .env (see docs/keycloak.md):
    UW sign-on   KC_UWYO_ISSUER, KC_UWYO_CLIENT_ID, KC_UWYO_CLIENT_SECRET   (OIDC)
    Google       KC_GOOGLE_CLIENT_ID, KC_GOOGLE_CLIENT_SECRET
    Microsoft    KC_MICROSOFT_CLIENT_ID, KC_MICROSOFT_CLIENT_SECRET
* outgoing mail, when KC_SMTP_HOST is set (with KC_SMTP_PORT, KC_SMTP_FROM, KC_SMTP_USER,
  KC_SMTP_PASSWORD, KC_SMTP_STARTTLS).

With --admin-email it also creates a realm account for that address (email marked
verified, no password) so its owner can set a password in the console. It never sets or
prints a password.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
BASE = os.environ.get("KC_INTERNAL_URL", "http://127.0.0.1:8180/sso")
REALM = "ailawlab"
CLIENT_ID = "ai-law-lab"
APP_URL = os.environ.get("AILAWLAB_PUBLIC_URL", "https://datahive.uwyo.edu/ai_law_lab").rstrip("/")


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.split(" #", 1)[0].strip()
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith(("KC_", "AILAWLAB_"))}}


def append_env(key: str, value: str, comment: str) -> None:
    text = ENV_FILE.read_text()
    if f"\n{key}=" in f"\n{text}":
        return
    with ENV_FILE.open("a") as f:
        f.write(f"{'' if text.endswith(chr(10)) else chr(10)}# {comment}\n{key}={value}\n")
    ENV_FILE.chmod(0o600)


class Admin:
    def __init__(self, env: dict[str, str]):
        if env.get("KC_ADMIN_USERNAME"):
            user = env["KC_ADMIN_USERNAME"]
            password = env.get("KC_ADMIN_PASSWORD") or (
                getpass.getpass(f"Keycloak console password for {user}: ") if sys.stdin.isatty() else "")
        else:
            user = env.get("KC_BOOTSTRAP_ADMIN_USERNAME") or "bootstrap-admin"
            password = env.get("KC_BOOTSTRAP_ADMIN_PASSWORD", "")
        if not password:
            sys.exit("No console admin password: set KC_ADMIN_USERNAME in .env and run this in a "
                     "terminal to be asked for it.")
        r = httpx.post(f"{BASE}/realms/master/protocol/openid-connect/token",
                       data={"grant_type": "password", "client_id": "admin-cli",
                             "username": user, "password": password}, timeout=30)
        if r.status_code != 200:
            sys.exit(f"Could not sign in to the Keycloak console as {user!r} (HTTP {r.status_code}). "
                     "If you replaced the bootstrap admin, put your admin's user name in .env as "
                     "KC_ADMIN_USERNAME and run this in a terminal.")
        self.http = httpx.Client(base_url=f"{BASE}/admin/realms", timeout=30,
                                 headers={"Authorization": f"Bearer {r.json()['access_token']}"})

    def get(self, path: str):
        r = self.http.get(path)
        return None if r.status_code == 404 else r.raise_for_status().json()

    def put(self, path: str, body) -> None:
        self.http.put(path, json=body).raise_for_status()

    def post(self, path: str, body) -> httpx.Response:
        r = self.http.post(path, json=body)
        r.raise_for_status()
        return r


def realm_settings(env: dict[str, str]) -> dict:
    smtp = env.get("KC_SMTP_HOST")
    body = {
        "realm": REALM, "enabled": True, "displayName": "AI Law Lab",
        "displayNameHtml": "AI Law Lab",
        "loginWithEmailAllowed": True, "duplicateEmailsAllowed": False,
        "registrationEmailAsUsername": True, "rememberMe": True, "editUsernameAllowed": False,
        # Self-service accounts need email: to verify the address, and to reset a password.
        "registrationAllowed": bool(smtp), "verifyEmail": bool(smtp),
        "resetPasswordAllowed": bool(smtp),
        "bruteForceProtected": True, "failureFactor": 8, "permanentLockout": False,
        "maxFailureWaitSeconds": 900, "waitIncrementSeconds": 60,
        "passwordPolicy": "length(12) and notUsername and notEmail and maxLength(128)",
        "sslRequired": "external",
        "ssoSessionIdleTimeout": 8 * 3600, "ssoSessionMaxLifespan": 24 * 3600,
        "accessTokenLifespan": 300,
    }
    if smtp:
        body["smtpServer"] = {
            "host": smtp, "port": env.get("KC_SMTP_PORT", "587"),
            "from": env.get("KC_SMTP_FROM", ""), "fromDisplayName": "AI Law Lab",
            "starttls": env.get("KC_SMTP_STARTTLS", "true"), "ssl": env.get("KC_SMTP_SSL", "false"),
            "auth": "true" if env.get("KC_SMTP_USER") else "false",
            "user": env.get("KC_SMTP_USER", ""), "password": env.get("KC_SMTP_PASSWORD", ""),
        }
    return body


def ensure_realm(kc: Admin, env: dict[str, str]) -> None:
    body = realm_settings(env)
    if kc.get(f"/{REALM}") is None:
        kc.http.post("", json=body).raise_for_status()
        print(f"created realm {REALM}")
    else:
        kc.put(f"/{REALM}", body)
        print(f"updated realm {REALM}")
    print(f"  self-registration and password reset by email: {'on' if body['registrationAllowed'] else 'off (no KC_SMTP_HOST)'}")


def ensure_client(kc: Admin) -> str:
    body = {
        "clientId": CLIENT_ID, "name": "AI Law Lab", "enabled": True, "protocol": "openid-connect",
        "publicClient": False, "standardFlowEnabled": True, "directAccessGrantsEnabled": False,
        "implicitFlowEnabled": False, "serviceAccountsEnabled": False,
        "redirectUris": [f"{APP_URL}/auth/callback"],
        "webOrigins": ["+"], "rootUrl": APP_URL, "baseUrl": f"{APP_URL}/",
        "attributes": {"pkce.code.challenge.method": "S256",
                       "post.logout.redirect.uris": f"{APP_URL}/*"},
        "protocolMappers": [{
            # Which identity provider someone signed in through ("uwyo", "google",
            # "microsoft"), or absent for a Keycloak account. The app uses it with the
            # verified email to decide who is approved automatically.
            "name": "identity provider", "protocol": "openid-connect",
            "protocolMapper": "oidc-usersessionmodel-note-mapper",
            "config": {"user.session.note": "identity_provider", "claim.name": "identity_provider",
                       "jsonType.label": "String", "id.token.claim": "true",
                       "access.token.claim": "false", "userinfo.token.claim": "true"},
        }],
    }
    found = kc.get(f"/{REALM}/clients?clientId={CLIENT_ID}") or []
    if found:
        cid = found[0]["id"]
        mappers = kc.get(f"/{REALM}/clients/{cid}/protocol-mappers/models") or []
        body_no_mappers = {k: v for k, v in body.items() if k != "protocolMappers"}
        kc.put(f"/{REALM}/clients/{cid}", {**found[0], **body_no_mappers})
        if not any(m["name"] == "identity provider" for m in mappers):
            kc.post(f"/{REALM}/clients/{cid}/protocol-mappers/models", body["protocolMappers"][0])
        print(f"updated client {CLIENT_ID}")
    else:
        kc.post(f"/{REALM}/clients", body)
        cid = kc.get(f"/{REALM}/clients?clientId={CLIENT_ID}")[0]["id"]
        print(f"created client {CLIENT_ID}")
    return kc.get(f"/{REALM}/clients/{cid}/client-secret")["value"]


PROVIDERS = [
    # alias, Keycloak provider, display name, env prefix, trust the email it reports
    ("uwyo", "oidc", "UW sign-on", "KC_UWYO", True),
    ("google", "google", "Google", "KC_GOOGLE", True),
    # A Microsoft work account's email address is set by whoever runs that tenant, so it
    # proves nothing about ownership; people who sign in this way are approved by an admin.
    ("microsoft", "microsoft", "Microsoft", "KC_MICROSOFT", False),
]


def ensure_providers(kc: Admin, env: dict[str, str]) -> None:
    for alias, provider, name, prefix, trust in PROVIDERS:
        client_id, secret = env.get(f"{prefix}_CLIENT_ID"), env.get(f"{prefix}_CLIENT_SECRET")
        if not (client_id and secret):
            print(f"  {name}: not configured ({prefix}_CLIENT_ID / {prefix}_CLIENT_SECRET)")
            continue
        config = {"clientId": client_id, "clientSecret": secret, "syncMode": "IMPORT",
                  "defaultScope": "openid email profile", "pkceEnabled": "true",
                  "pkceMethod": "S256"}
        if provider == "oidc":
            issuer = env.get(f"{prefix}_ISSUER", "").rstrip("/")
            if not issuer:
                print(f"  {name}: set {prefix}_ISSUER too (the sign-on service's issuer URL)")
                continue
            meta = httpx.get(f"{issuer}/.well-known/openid-configuration", timeout=30).raise_for_status().json()
            config.update({"issuer": meta["issuer"], "authorizationUrl": meta["authorization_endpoint"],
                           "tokenUrl": meta["token_endpoint"], "jwksUrl": meta["jwks_uri"],
                           "userInfoUrl": meta.get("userinfo_endpoint", ""),
                           "logoutUrl": meta.get("end_session_endpoint", ""),
                           "validateSignature": "true", "useJwksUrl": "true",
                           "clientAuthMethod": "client_secret_post"})
        if provider == "microsoft":
            # "common": work or school accounts from any organization, and personal accounts.
            config["tenantId"] = env.get(f"{prefix}_TENANT", "common")
        body = {"alias": alias, "providerId": provider, "displayName": name, "enabled": True,
                "trustEmail": trust, "storeToken": False, "firstBrokerLoginFlowAlias": "first broker login",
                "config": config}
        if kc.get(f"/{REALM}/identity-provider/instances/{alias}") is None:
            kc.post(f"/{REALM}/identity-provider/instances", body)
            print(f"  {name}: added")
        else:
            kc.put(f"/{REALM}/identity-provider/instances/{alias}", body)
            print(f"  {name}: updated")


def ensure_user(kc: Admin, email: str) -> None:
    found = kc.get(f"/{REALM}/users?email={email}&exact=true") or []
    if found:
        print(f"realm account for {email} already exists")
        return
    kc.post(f"/{REALM}/users", {"username": email, "email": email, "emailVerified": True,
                                "enabled": True})
    print(f"created a realm account for {email} (no password yet: set one in the console, "
          "Users > this user > Credentials)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admin-email", help="create a realm account for this address")
    args = ap.parse_args()
    env = load_env()
    kc = Admin(env)
    ensure_realm(kc, env)
    secret = ensure_client(kc)
    append_env("AILAWLAB_OIDC_CLIENT_SECRET", secret,
               "Secret of the ai-law-lab client in Keycloak (written by scripts/keycloak_setup.py)")
    print("identity providers:")
    ensure_providers(kc, env)
    if args.admin_email:
        ensure_user(kc, args.admin_email)


if __name__ == "__main__":
    main()
