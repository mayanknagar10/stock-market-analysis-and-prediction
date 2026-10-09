# Prospective automation and backend phase report
Date: 2026-10-09. Branch: research/prospective-automation. Recoverable research baseline: b3a9edc. Implementation checkpoint: b227ae5 (63files changed). This report is finalized in a subsequent documentation checkpoint.

The engineering phase is complete. The first real canonical window is still prospective: origin2026-10-09, capture2026-10-10T00:30–01:00UTC and cutoff01:00UTC (06:30IST). No canonical batch was fabricated or backfilled. The original388manual forecasts remain a separate origin group.

## Preservation
Verified1,052original V5 files,2,904additional research files and all388original forecast payload hashes. Frozen candidate research-20261008T190859, consumed-final locks/markers, failed gate, quarantined candidates, source snapshots, V4/V5 comparison and historical conclusions remain unchanged. Existing forecasts remain LOW/abstained; production remains blocked.

| Change | Answer |
|---|---|
| Frozen model/inference/feature logic changed | NO |
| Model parameters, calibration/confidence/OOD/abstention thresholds changed | NO |
| Successor final candidate trained | NO |
| Consumed V5 final period reused for decisions | NO |
| Production model enabled | NO |
| Next.js frontend started | NO |

New work changes orchestration, input admission, provenance, API contracts and scheduling.

## Architecture and files
- application/collection: official2026calendar, deterministic store/collector, pure frozen-native engine and independent resolver.
- application/services: shared read-only ledger/validation, data access/financial analytics, portfolio boundary, saved Streamlit workspace and Insights facade.
- application/schemas.py and api/main.py: Pydantic contracts and21versioned readGETroutes. No request path fits models, calibrates, runs walk-forward research or captures forecasts.
- application/provenance.py: actual issuing-code/helper/config fingerprints, native artifact/manifests/parameters and feature-vector hashes.
- jobs/: capture/finalize, exact maturity resolution, timezone-independent dispatcher and append-only provenance audit.
- pages/forecast.py, pages/insights.py, pages/research_results.py: shared saved services; no prediction issuance from page views/questions.
- config/collection/calendar_sources: exact official NSE source; source hash checked.
- scripts/: preservation verification, OpenAPI export, reproducible latency measurement and Windows scheduler support.
- docs/: collection protocol, scheduling, API contract, deployment, this report and updated master task board.
- reports/automation/: preservation manifest, live/fixture measurements, scheduler state, independent review and timestamp audit.
- tests/:38new automation tests; existing resolver fixtures now write isolated test reports rather than genuine research paths.

## Collection and idempotency
The frozen100stock/17industry universe is enforced by byte hash. Canonical origin is delayed after-close06:30IST on D+1. Capture06:00–06:30; inference/admission06:30–07:00. Official holidays/special sessions govern eligibility; unregistered years and uncertain Muhurat timings fail closed.

UUIDv5 identities derive from protocol/origin/model/universe and ticker/horizon. OS writer locks exclude concurrent jobs. Attempts are logged before provider calls, capped at3 across crashes; delay/cutoff never slides. Content-addressed snapshots and prediction/outcome/batch payloads are immutable. Atomic four-horizon stock writes permit recovery of missing stocks only, against already frozen inputs. Every member retains a success/failure/skipped status. Recovered provider failures remain visible. Sealed reruns verify hashes and append only separate duplicate-prevention events.

## Maturity and validation
Official exact1/5/10/20session endpoints are resolved independently. No missing candle shifts maturity. Stock/benchmark/sector/fixed-peer receipts must precede effective resolution and their own completed endpoint. Vendor adjusted-close total-return ratios, raw final prices, actions/revisions, Nifty price-index convention, sector policy and alpha/error/coverage are explicit.

There are now97matured legacy1D outcomes and291pending forecasts; no canonical forecasts have yet been issued. Group separation, final-boundary masking, raw-versus-calibrated probability semantics and counts prevent retrospective/forward pooling. Forward state remains INSUFFICIENT_FORWARD_EVIDENCE. Review sufficiency requires each horizon's count/stock/industry/regime/origin/data-quality criteria; it cannot authorize promotion.

The first resolver used a batch-start timestamp before acquisition for77outcomes. Those immutable outcomes were retained. Hash-bound append-only amendments record later audited acceptance without changing predictions, return values or source vintages. Independent verification found zero discrepancies in stock/benchmark/sector/peer values and clocks. Unamended invalid ordering is excluded. The other20outcomes used the corrected resolver. All97are valid after audit. The failed initial run remains visible in INITIAL_OUTCOME_TIMESTAMP_AUDIT.json and REVIEW.md.

## Tests and review
Full regression:203passed,9warnings;165previous tests maintained plus38new tests. Final targeted collection/scheduling run:18passed. API, typed schemas, saved UI and legacy workspace/scenarios passed. Independent review has no unresolved Critical/Important findings. Original V5/research/prediction preservation reverified.

Warnings comprise existing UI utcnow deprecations and a Starlette/httpx TestClient deprecation. They do not represent a failed test or a model upgrade.

## API and performance
All requested system/model/validation/stock/forecast/market/data/collection routes are implemented under /api/v1; additional risk/factors/screener/performance routes bring the total to21. Forecast/history contracts expose research-only and recommendation_allowed=false, awareUTC timestamps, provenance, abstention, calibrated/null versus raw probability and separately realized outcomes. Structured errors omit paths/tracebacks. Private snapshots and feature vectors are not public endpoints.

OpenAPI: docs/api/openapi.json. API_CONTRACT.md documents generated TypeScript types, pagination, auth, units/nulls and caching. Successful provider payloads have a30-second128key cache; errors are not cached as successes. No indefinite quote/forecast cache is used.

Bounded live measurements (5requests/route, cold plus warm):
| Operation | Median ms | p95 ms |
|---|---:|---:|
| Provider history |236.4|1130.0|
| Stock overview |4.3|990.5|
| History |272.0|292.6|
| Technicals |125.3|130.8|
| Saved legacy forecast |366.7|375.3|
| Validation summary |3.4|3.9|
| Market overview |7.8|319.9|
| Frozen native inference replay,20samples |27.9|45.0|

All live route samples returned200; native central-return replay matched saved forecasts. These small measurements are not capacity guarantees. The first synthetic-provider timing run failed its fixture timezone contract and remains preserved; corrected V2fixture measurements succeeded. Saved HTTP forecast requests perform no inference.

## Scheduling and deployment
Installed local task StockPro-Prospective-Research-v1: enabled, five-minute dispatch, overlap ignored, battery allowed, exit0 verified through actual scheduler logs. It requires an awake PC and current-user login. Its first capture is upcoming; uninterrupted hosting has not been selected.

Standalone commands:
    python -m jobs.capture_prospective_forecasts --stage capture
    python -m jobs.capture_prospective_forecasts --stage finalize
    python -m jobs.resolve_matured_outcomes
    python -m jobs.scheduled_tick

Local API:
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Health: /api/v1/health. Streamlit remains on8501. Deployment-neutral cron/external/local/self-hosted scheduling is documented. GitHub schedules can be late; the job never backdates.

Python3.12.10 and the tested native/backend dependencies are required. Persist private data/prospective_automation and existing research/model artifacts across redeploys. SQLite/OS locks assume one durable writer. Configure STOCKPRO_DATA_ROOT, optional bearer token/CORS and private timing logs. Public exposure needs its own authenticated/TLS/rate-limited deployment.

## Remaining limitations
The first genuine canonical batch and long-term evidence require real time. The PC/login dependency prevents always-on guarantees. Calendar2027and pending special timings require official versioned registration. Vendor data and historical training vintages remain imperfect; no PIT training certification is claimed. Unsupported symbol changes/delistings remain visible failures/pending states. Daily full-history snapshots need storage monitoring and durable backup. Validation currently scans immutable ledgers; measure/index aggregate storage before larger deployment. Browser visual/accessibility QA was not available. Production promotion and a successor final candidate remain blocked. Hosting and full Next.js migration are separate tasks.
