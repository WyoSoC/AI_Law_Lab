# Accounts: Keycloak sign-in for AI Law Lab

People sign in through **Keycloak**, a self-hosted sign-in service, with UW sign-on, Google,
Microsoft, or an AI Law Lab account of their own. Keycloak proves who someone is; the app
decides what they may do (`src/ailawlab/accounts.py`).

| | |
|---|---|
| Keycloak | container `ailawlab-keycloak` (`docker-compose.yml`), 127.0.0.1:8180, image pinned to 26.7.5 |
| Public address | `https://datahive.uwyo.edu/sso/` (Apache proxy; `/auth` and `/admin` belong to Open WebUI) |
| Realm / client | `ailawlab` / `ai-law-lab` |
| Keycloak's data | database `keycloak` in the `ailawlab-db` Postgres |
| Configuration | `scripts/keycloak_setup.py`, re-runnable, reads `.env` |
| Console | `https://datahive.uwyo.edu/sso/admin/` |

## Who gets in

On a first sign-in (`accounts.initial_access`):

- an address in `AILAWLAB_ADMIN_EMAILS` becomes an **admin**, if the provider vouches for it;
- a verified `@uwyo.edu` address (`AILAWLAB_AUTO_APPROVE_DOMAINS`) becomes a **researcher**;
- everyone else is **pending** until an admin approves them on the **Users** page.

Addresses reported by **Microsoft are never trusted** for this: the address on a work or school
account is set by whoever runs that organization's tenant, so it proves nothing (the "nOAuth"
problem). Microsoft users are approved by an admin; UW people are auto-approved when they use UW
sign-on or Google.

Roles: **viewer** reads and exports; **researcher** also creates experiments, runs them and curates
libraries; **admin** also manages people, empties the trash, deletes libraries, and deletes any
experiment for good (researchers may delete their own). Everything is shared lab-wide. Approvals,
role changes and deletions are written to `audit_log` and shown on the Users page.

## One-time setup

1. **Route `/sso/` through Apache.** Add this beside the AI Law Lab block in
   `/etc/apache2/sites-available/ood-portal.conf` (hand-edited; do not run `update_ood_portal`),
   then `sudo apache2ctl configtest && sudo systemctl reload apache2`:

   ```apache
   # Keycloak sign-in service for AI Law Lab (ailawlab-keycloak on 127.0.0.1:8180).
   RedirectMatch 301 ^/sso$ /sso/
   ProxyPass /sso/ http://127.0.0.1:8180/sso/ timeout=60
   ProxyPassReverse /sso/ http://127.0.0.1:8180/sso/
   <Location /sso/>
     RequestHeader set X-Forwarded-Proto "https"
     RequestHeader set X-Forwarded-Port "443"
   </Location>
   ```

2. **Replace the temporary console admin.** Sign in at `/sso/admin/` as `bootstrap-admin` with
   `KC_BOOTSTRAP_ADMIN_PASSWORD` from `.env`. In the *master* realm: Users → Add user (your
   name) → Credentials → set a password → Role mapping → assign `admin`. Sign in as yourself,
   delete `bootstrap-admin`, and set up two-factor for your account (Account console → Signing
   in → Authenticator application). Then put `KC_ADMIN_USERNAME=<you>` in `.env` and remove the
   `KC_BOOTSTRAP_ADMIN_*` lines; the setup script will ask for your password when it runs.

3. **Your lab account.** Set `AILAWLAB_ADMIN_EMAILS=you@uwyo.edu` in `.env`. Either sign in with
   UW sign-on or Google once they are connected, or create a realm account now:
   `.venv/bin/python scripts/keycloak_setup.py --admin-email you@uwyo.edu`, then in the console
   switch to the *ailawlab* realm → Users → that user → Credentials → Set password.

## Connecting sign-in providers

Each provider needs an app registration with this **redirect URI**, then its credentials in
`.env`, then `scripts/keycloak_setup.py` again. Nothing else changes: Keycloak shows a button for
each connected provider on its sign-in page.

| Provider | Redirect URI | `.env` |
|---|---|---|
| UW sign-on | `https://datahive.uwyo.edu/sso/realms/ailawlab/broker/uwyo/endpoint` | `KC_UWYO_ISSUER`, `KC_UWYO_CLIENT_ID`, `KC_UWYO_CLIENT_SECRET` |
| Google | `https://datahive.uwyo.edu/sso/realms/ailawlab/broker/google/endpoint` | `KC_GOOGLE_CLIENT_ID`, `KC_GOOGLE_CLIENT_SECRET` |
| Microsoft | `https://datahive.uwyo.edu/sso/realms/ailawlab/broker/microsoft/endpoint` | `KC_MICROSOFT_CLIENT_ID`, `KC_MICROSOFT_CLIENT_SECRET` |

- **UW sign-on**: ask UW IT for an OpenID Connect client ("web application", scopes
  `openid email profile`) with the redirect URI above. They supply the issuer URL (for Microsoft
  Entra ID, `https://login.microsoftonline.com/<UW tenant id>/v2.0`), client ID and secret. If
  UW IT can only offer SAML (Shibboleth/InCommon), Keycloak supports that too, but the setup
  script covers OIDC only; the provider can be added in the console instead.
- **Google**: Google Cloud Console → a project → APIs & Services → OAuth consent screen
  (External, app name "AI Law Lab", scopes openid/email/profile) → Credentials → Create OAuth
  client ID → Web application, with the redirect URI. Basic scopes need no Google review, but
  Google may ask you to verify the domain (`uwyo.edu`); if so, UW IT can help.
- **Microsoft**: Microsoft Entra admin center (any tenant you can register apps in) → App
  registrations → New → "Accounts in any organizational directory and personal Microsoft
  accounts" → Web redirect URI as above → Certificates & secrets → New client secret.
  Secrets expire (at most two years): note the date and replace it before then. Some
  organizations block their staff from consenting to outside apps; their IT must allow it.

## Email (for AI Law Lab accounts)

Self-registration, email verification and password resets are **off** until Keycloak can send
mail. To turn them on, set `KC_SMTP_HOST`, `KC_SMTP_PORT` (587), `KC_SMTP_FROM`, and if the
server needs a login `KC_SMTP_USER` / `KC_SMTP_PASSWORD` (UW's relay, or a lab mailbox with an app
password), then run the setup script. Until then, admins can create accounts in the console.

## Reaching the site from off campus

`datahive.uwyo.edu` resolves publicly to a private address (172.26.7.78), so only campus and VPN
users can reach the lab or its sign-in page. External users need UW IT to publish the site.

## Running it

- Start / restart: `docker compose -p ai_law_lab up -d keycloak` (from the repository).
- A brand-new install has no console admin. For its first start only, add
  `KC_BOOTSTRAP_ADMIN_USERNAME` and `KC_BOOTSTRAP_ADMIN_PASSWORD` to the keycloak service's
  `environment` (with values), start it, replace that admin as in step 2, then remove both lines
  again. Keycloak refuses to start if the username is set but the password is empty.
- Logs: `docker logs ailawlab-keycloak`.
- Backups: Keycloak's data is the `keycloak` database in the same Postgres volume as the lab;
  back up both, e.g. `docker exec ailawlab-db pg_dump -U ailawlab keycloak > keycloak.sql`.
- Upgrades: read Keycloak's upgrade notes, change the image tag in `docker-compose.yml`, and
  `docker compose -p ai_law_lab up -d keycloak`; it migrates its database on start.
- Local development without sign-in: `AILAWLAB_AUTH_REQUIRED=false` (every request then acts as an
  admin). Never set it on the server.
