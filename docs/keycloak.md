# Accounts: Keycloak sign-in for AI Law Lab

People sign in through **Keycloak**, a self-hosted sign-in service, with UW sign-on or an AI Law
Lab account of their own (people outside UW request one on the sign-in page). Keycloak proves who someone is; the app
decides what they may do (`src/ailawlab/accounts.py`).

| | |
|---|---|
| Keycloak | container `ailawlab-keycloak` (`docker-compose.yml`), 127.0.0.1:8180, image pinned to 26.7.5 |
| Public address | `https://datahive.uwyo.edu/sso/` (Apache proxy; `/auth` and `/admin` belong to Open WebUI) |
| Realm / client | `ailawlab` / `ai-law-lab` |
| Keycloak's data | database `keycloak` in the `ailawlab-db` Postgres |
| Configuration | `scripts/keycloak_setup.py`, re-runnable, reads `.env` |
| Sign-in theme | `keycloak/themes/ailawlab` (UW sign-on first, then an AI Law Lab account, a link to request an account, and the terms acknowledgement) |
| Console | `https://datahive.uwyo.edu/sso/admin/` |

## Who gets in

On a first sign-in (`accounts.initial_access`):

- an address in `AILAWLAB_ADMIN_EMAILS` becomes an **admin**, if the provider vouches for it;
- a verified `@uwyo.edu` address (`AILAWLAB_AUTO_APPROVE_DOMAINS`) becomes a **researcher**;
- everyone else is **pending**. After signing in they see a short **access request** form: name,
  organization, what they want to use the lab for, and whether they need viewer or researcher
  access (nobody can ask to be an admin). The request appears under **Access requests** on the
  **Users** page, with a count beside *Users* in the menu. An admin approves it (with the role
  asked for, or another) or declines it with an optional note, which the person sees the next
  time they sign in. People who signed in but never sent the form are listed separately and
  can be approved or declined too. A declined person can be let in later with **Enable**.

With an outgoing mail server (see *Email* below), admins are emailed when a request arrives and the
person is emailed the decision (`src/ailawlab/notify.py`). Without one, nothing is sent: check the
Users page.

Only UW sign-on and Keycloak's own accounts count as proof of an address
(`accounts.TRUSTED_EMAIL_PROVIDERS`); UW people are auto-approved when they use UW sign-on. Google
and Microsoft sign-in were offered once and are no longer: the setup script removes them from the
realm if it finds them, and an address from any provider not on that list is never trusted.

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
   UW sign-on once it is connected, or create a realm account now:
   `.venv/bin/python scripts/keycloak_setup.py --admin-email you@uwyo.edu`, then in the console
   switch to the *ailawlab* realm → Users → that user → Credentials → Set password.

## Connecting UW sign-on

UW sign-on is the one sign-in provider (`KC_UWYO_SAML` in `.env`).

- **UW sign-on** is SAML with UW's Shibboleth (`https://shibboleth.uwyo.edu/idp/shibboleth`),
  which is what UW IT prefers. The setup script reads UW's metadata from InCommon
  (`mdq.incommon.org`; shibboleth.uwyo.edu itself refuses connections from datahive), and
  Keycloak keeps re-reading it, so UW's certificate changes need nothing from us.
  1. Set `KC_UWYO_SAML=hidden` and run the setup script. This creates the provider without a
     button on the sign-in page, and publishes our metadata at
     `https://datahive.uwyo.edu/sso/realms/ailawlab/broker/uwyo/endpoint/descriptor`
     (entity ID `https://datahive.uwyo.edu/sso/realms/ailawlab`). UW IT registers that.
  2. Ask UW IT to release `eduPersonPrincipalName` (Keycloak's identifier for the person),
     `mail`, `givenName` and `sn`. Without eduPersonPrincipalName the sign-in fails.
  3. Once they confirm, set `KC_UWYO_SAML=on` and run the script again to show the button.
  Our signing and encryption keys are the realm's own; if they are ever rotated (Realm settings →
  Keys), tell UW IT, since they may have stored a copy of the metadata.

## Email (for AI Law Lab accounts)

People outside UW request an account from the sign-in page ("Not at UW? Request an account"),
which is Keycloak's registration form: email, password, name. They arrive **pending** and fill in
the access request; nothing works until an admin approves it. A `@uwyo.edu` address is turned
away on that form and sent to UW sign-on, since an account made there would block that person's
own UW sign-on later (the Users page also flags any unverified address). Email verification and
password resets stay **off** until Keycloak can send mail, and so do the app's access-request
emails (it uses the same `KC_SMTP_*` settings; restart the app after setting them). To turn them on, set `KC_SMTP_HOST`, `KC_SMTP_PORT` (587), `KC_SMTP_FROM`, and if the
server needs a login `KC_SMTP_USER` / `KC_SMTP_PASSWORD` (UW's relay, or a lab mailbox with an app
password), then run the setup script. Until then, admins can create accounts in the console.

## Running it

- Start / restart: `docker compose -p ai_law_lab up -d keycloak` (from the repository).
- A brand-new install has no console admin. For its first start only, add
  `KC_BOOTSTRAP_ADMIN_USERNAME` and `KC_BOOTSTRAP_ADMIN_PASSWORD` to the keycloak service's
  `environment` (with values), start it, replace that admin as in step 2, then remove both lines
  again. Keycloak refuses to start if the username is set but the password is empty.
- Sign-in pages: the realm's login theme is `ailawlab`, Keycloak's own `keycloak.v2` with the
  sign-in page reordered (`login.ftl`: UW sign-on first, then the AI Law Lab account form, then
  "Request an account"), its wording in `messages/messages_en.properties`, and the Terms of Use
  acknowledgement under the box, "By signing in…" or "By creating an account…" (`footer.ftl`;
  styles in `resources/css/ailawlab.css`). `docker-compose.yml` mounts it read-only. Keycloak caches
  themes, so after editing it run `docker compose -p ai_law_lab up -d --force-recreate keycloak`.
  Browsers keep the theme's files for 30 days at an address that does not change, so after
  editing `ailawlab.css` also change its `?v=` tag in `theme.properties`.
- Logs: `docker logs ailawlab-keycloak`.
- Backups: Keycloak's data is the `keycloak` database in the same Postgres volume as the lab;
  back up both, e.g. `docker exec ailawlab-db pg_dump -U ailawlab keycloak > keycloak.sql`.
- Upgrades: read Keycloak's upgrade notes, change the image tag in `docker-compose.yml`, and
  `docker compose -p ai_law_lab up -d keycloak`; it migrates its database on start.
- Local development without sign-in: `AILAWLAB_AUTH_REQUIRED=false` (every request then acts as an
  admin). Never set it on the server.
