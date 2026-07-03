# data/

Local, zero-external-account storage for this app.

## `users.json`
Login credentials (username, salted+hashed password, display name, role).
Created automatically on first run with a demo account (`demo` / `demo1234`).

**Persistence on Streamlit Cloud:** the free tier's filesystem is
ephemeral — self-registered accounts are lost on redeploy/restart.
Two ways to handle this:
- **Quick:** commit `data/users.json` to git after registering the
  accounts you want to keep. They'll persist across every redeploy,
  same pattern as the ML checkpoint in `models/`.
- **Durable:** swap to a real database (Supabase/Neon Postgres) when
  you're ready — only `core/auth.py`'s `verify_login()` / `create_user()`
  functions need to change.

Passwords are hashed with `hashlib.pbkdf2_hmac` (260,000 iterations,
random salt per user) — never stored in plaintext, never logged.

## `cache/`
Disk-backed cache (via the `diskcache` package) for API responses —
speeds up repeat requests and survives app restarts (as long as the
underlying disk survives — see caveat above, same ephemeral-filesystem
caveat applies). Gitignored — this is regenerated automatically, never
needs to be committed.

Safe to delete at any time; the app will just refetch from the network
on the next request.
