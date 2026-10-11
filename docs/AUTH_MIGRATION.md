# Authentication migration

## Current boundary

The new web platform does not import core/auth.py or expose data/users.json. The retained Streamlit console uses a local JSON account store, salted PBKDF2 hashes and Streamlit session state. Its module seeds a demo account on import. Those local accounts, sessions and demo credentials are not web identities and are not accepted by FastAPI. The older module's suggestion to commit seeded accounts must not be used for this migration. No user-store contents were read or published during this audit.

Public research GETs can be protected with STOCKPRO_API_TOKEN. That shared token is a read-access guard, not a user login or operator authorization. Do not embed it in NEXT_PUBLIC variables. A hosted frontend requiring protected reads will need a separately approved server-side read proxy or identity integration.

Sensitive stateless analyses, scenarios and manual jobs require separate operator authentication. Local development explicitly enables a loopback-only session: STOCKPRO_ENABLE_LOCAL_OPERATOR=1, STOCKPRO_ENV=development, exact CORS Origin, one API worker. Forwarded headers are rejected. HttpOnly/SameSite=Strict cookies expire after one hour and are held in server memory; restarting the backend expires them. No cookie secret is returned in JSON. Production disables local session provisioning.

A private remote operator bearer is configured with STOCKPRO_OPERATOR_TOKEN and must differ from STOCKPRO_API_TOKEN. Any browser-entered operator bearer must stay in memory for that tab/session; never persist it in localStorage, URLs, analytics, public environment variables or logs. This is private operator access, not a multi-user hosted authentication system.

## Guest data

Portfolio/watchlist inputs are browser-local guest data. They are submitted only to authenticated, bounded Python analysis endpoints. No broker connection, server account persistence or execution is provided. Existing Streamlit user data are not silently copied to the browser.

## Controlled hosted migration

Before public sensitive workflows, choose a durable identity/session store and ownership model separately from Python hosting. Define account migration consent, password reset/re-enrollment, access roles, CSRF/origin protection, cookie domains/TLS, session revocation, rate limits and audit retention. Remove any demo-account dependency from the hosted path. Import no raw password hashes into frontend assets and expose no arbitrary file access.

Until that work is verified, deploy research reads only as appropriate for their configured access policy. Sensitive workflows fail closed without authorized operator credentials. This phase does not claim complete hosted account authentication.
