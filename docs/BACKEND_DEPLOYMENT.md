# Backend deployment assumptions
This phase prepares the Python backend/scheduler; it does not choose hosting or start Next.js.

## Runtime
Validated on CPython3.12. Install requirements.txt in an isolated environment; native model dependency upgrades require compatibility verification against frozen artifacts.
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
    python -m streamlit run app.py --server.port 8501

GET /api/v1/health is process liveness, not evidence/provider reliability. Inspect model/health, data/health and collection/status separately.

## Configuration
| Variable | Meaning |
|---|---|
| STOCKPRO_DATA_ROOT | Durable private canonical data; default data/prospective_automation |
| STOCKPRO_API_TOKEN | Optional bearer token; protects all routes except liveness |
| STOCKPRO_CORS_ORIGINS | Comma-separated exact browser origins; empty default |
| STOCKPRO_PERFORMANCE_LOG | Optional private JSONL timing log |
| PYTHONPATH | Local .deps support; omit in normal installed environment |

Bind loopback by default. Public exposure requires TLS, authentication and rate limiting. No training/mutation endpoint exists. Swagger/OpenAPI also follow token middleware. Raw snapshots/data directories must never be public/static files. Credentials never enter TypeScript or artifacts. Future portfolio/user identity integration is separate.

## Persistence
Retain frozen models/v5, ACTIVE_RESEARCH.json, original reports/config/calendar, all data/research archives and the388original forecast payloads. Canonical collection.sqlite stores separate immutable forecasts/outcomes/batches plus controls/events. archive/ retains content-addressed raw normalized payloads. Retain resolver markers and operational/performance logs across redeploys.

Jobs require writable persistent storage; API reads use SQLite read-only connections and do not issue forecasts. SQLite plus an OS lock assumes one durable host/writer. Ephemeral serverless filesystems, horizontally scaled writers and shared network filesystems require a separately designed storage/locking deployment; do not assume compatibility.

Monitor backups/restore, free space and clock synchronization. Daily119-source five-year snapshots can consume substantial disk; measure growth and budget capacity. No automatic deletion/retention rewrites are enabled.

## Scheduler and calendar
Use SCHEDULING.md. The local task requires an awake machine and logged-in user. Continuous operation needs an independently selected durable host. Calendar2026 is registered; unknown later years and Muhurat timing remain blocked until official registrations. Provider failures and100%abstention remain valid reported outcomes.

Before redeploy verify the preservation script, tests and generated OpenAPI. Evidence sufficiency cannot enable production. Backend state remains RESEARCH_ONLY, promotionBLOCKED and recommendation_allowed=false.
