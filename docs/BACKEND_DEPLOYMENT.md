# Backend deployment assumptions
The web platform keeps Python hosting independent of the Next.js frontend. No hosting service, deployment or GitHub work is activated. The approved development mode uses a local PC backend and durable local storage.

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
| STOCKPRO_OPERATOR_TOKEN | Separate private operator bearer for analyses/scenarios/manual jobs; differs from read token |
| STOCKPRO_ENABLE_LOCAL_OPERATOR | Explicit development-only loopback session opt-in (1) |
| STOCKPRO_ENV | production/prod disables local operator provisioning |
| STOCKPRO_CORS_ORIGINS | Comma-separated exact browser origins; empty default |
| STOCKPRO_PERFORMANCE_LOG | Optional private JSONL timing log |
| PYTHONPATH | Local .deps support; omit in normal installed environment |

Bind loopback by default. Public exposure requires TLS, authentication and rate limiting. No training or promotion endpoint exists. Sensitive stateless analysis/scenario POSTs and explicit asynchronous job POSTs are separately authenticated. Normal reads never invoke collection. Swagger/OpenAPI also follow token middleware. Raw snapshots/data directories must never be public/static files. Credentials never enter TypeScript or artifacts. Hosted user identity integration remains separate; browser-local guest analytics do not migrate the old account store.

## Persistence
Retain frozen models/v5, ACTIVE_RESEARCH.json, original reports/config/calendar, all data/research archives and the388original forecast payloads. Canonical collection.sqlite stores separate immutable forecasts/outcomes/batches plus controls/events. archive/ retains content-addressed raw normalized payloads. Retain resolver markers and operational/performance logs across redeploys.

Jobs require writable persistent storage; API reads use SQLite read-only connections and do not issue forecasts. SQLite plus an OS lock assumes one durable host/writer. Ephemeral serverless filesystems, horizontally scaled writers and shared network filesystems require a separately designed storage/locking deployment; do not assume compatibility.

Monitor backups/restore, free space and clock synchronization. Daily119-source five-year snapshots can consume substantial disk; measure growth and budget capacity. No automatic deletion/retention rewrites are enabled.

## Scheduler and calendar
Use SCHEDULING.md. The local task requires an awake machine and logged-in user. Continuous operation needs an independently selected durable host. Calendar2026 is registered; unknown later years and Muhurat timing remain blocked until official registrations. Provider failures and100%abstention remain valid reported outcomes.

Before redeploy verify the preservation script, tests and generated OpenAPI. Evidence sufficiency cannot enable production. Backend state remains RESEARCH_ONLY, promotionBLOCKED and recommendation_allowed=false.

## Web-platform operator runtime

Use one API worker for memory-only local sessions and a single durable research writer. Operator execution records require writable private data/web_platform/jobs storage in addition to canonical/legacy archives. Bounded CLI/provider subprocesses need operating-system process support; Windows launches suppress console windows. Local development helper services bind 127.0.0.1:8001 (API) and 127.0.0.1:3000 (frontend); deployment ports are configurable independently.

Native XGBoost/LightGBM/SHAP dependencies, model memory, cold starts, subprocess capability and persistent ledger/snapshot writes must be evaluated before choosing Python hosting. A frontend deployment on Vercel does not make the backend or scheduler durable. Do not force this research writer onto ephemeral serverless filesystems.
