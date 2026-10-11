# StockPro API v1
FastAPI delegates to shared services and existing financial calculations. Requests retrieve saved forecasts or perform descriptive calculations; normal reads never train, collect prediction inputs, reconstruct research datasets or calibrate. Explicit authenticated operator job submissions invoke the unchanged standalone collection/resolution commands asynchronously.

## Schema / TypeScript
    python scripts/export_openapi.py
The generated docs/api/openapi.json is the source of truth. In the separate frontend project:
    cd frontend
    npm run generate:api
Pin the generator there. Do not maintain competing handwritten interfaces. Review breaking schema changes in CI. Forecast/research responses have explicit Pydantic fields; descriptive technical/health maps retain dynamic fields.

## GET routes
All routes use /api/v1:
| Area | Routes |
|---|---|
| System | /health, /system/status, /system/performance |
| Model | /model/status, /model/health |
| Validation | /validation/summary, /validation/prospective |
| Stock | /stocks/search; /stocks/{ticker}/overview, /history, /technicals, /risk, /factors |
| Saved forecast | /stocks/{ticker}/forecast, /forecast/history, /forecast/explanation |
| Markets | /markets/overview, /markets/sectors |
| Operations | /data/health, /collection/status |
| Screening | /screener |

Search matches frozen Indian ticker/company metadata. Stock data support plainUS/.NS/.BO equities; reference indices/FX/futures use market services. Supported ticker data do not imply a saved forecast exists. Canonical is the default forecast/validation group. group=legacy explicitly selects the388manual intraday records; groups never pool. Historical reconstructed reports remain separate in the research UI.

## Additive web-platform services

The schema currently describes 41 paths / 44 HTTP operations; the original 21 GET endpoints remain compatible. Typed adapters add fundamentals, news, exchange-session OHLCV series, grouped technical series, risk reports, factor reports and market context. Saved research forecast detail, mature analogues and preserved V5 historical reports are separate from current provider vintages. No GET generates a new forecast.

Operator-protected stateless POSTs: /analysis/portfolio, /analysis/watchlist, /analysis/compare, /analysis/screen and /research/forecasts/{prediction_id}/scenario. Pydantic bounds request counts, symbols, dates, finite numbers and shock sizes; these calculations do not update accounts, source snapshots or frozen predictions. Portfolio values describe a hypothetical daily constant-weight analysis, with failed holdings and currency constraints explicit. Scenario estimates use cloned archived inputs and never become new ledger forecasts.

Operator routes: POST/GET/DELETE /operations/session; GET /operations/status; POST/GET /operations/jobs; GET /operations/jobs/{run_id}. POST jobs returns 202 with a typed JobRecord. The only job types are capture, finalize, resolve and verify. Capture/finalize require an ISO origin; canonical time/calendar guards remain authoritative. Terminal records are immutable. No arbitrary command, path or training route exists.

Local operator cookies require explicitly enabled development mode, loopback peer, exact configured Origin and absence of proxy-forwarding headers. The cookie is HttpOnly/SameSite=Strict, has one-hour server expiry and is never returned in JSON. Remote/private operator bearer authentication uses STOCKPRO_OPERATOR_TOKEN, which must differ from STOCKPRO_API_TOKEN. A shared read token does not authorize sensitive POSTs. Full hosted account authentication remains a separate migration.

## Requests/errors
Read requests use paths/queries, no mutation bodies. Search accepts q, offset>=0, limit1..1000. History/forecast-history/screener expose items/total/offset/limit. Validation groups retain sample counts.

Errors use {error:{code,message,request_id}} and X-Request-ID. Invalid ticker/request:422; no saved forecast:404; provider/data/integrity failure:503. Abstention is valid forecast semantics, not a fabricated failure/action. Pending outcomes are never zero-imputed. Responses omit private paths, tracebacks and credentials.

STOCKPRO_API_TOKEN requires Authorization: Bearer <token> except /health. Configure exact CORS origins. TLS/rate limits/identity management precede external exposure.

## Time/units/nulls
Timestamps are aware ISO8601, UTC offsets explicit. Indian sessions use Asia/Kolkata. Daily provider labels are session observations, not exact trade/publication time. Canonical cutoff06:30IST D+1 differs from the stored price session timestamp.

Returns are fractions:0.01=1%. Market context change_percent and watchlist percent fields explicitly declare units=percent (1=1 percentage point). Fundamentals dividend_yield is normalized to a fraction only when the provider unit is verified; otherwise it is null with separate raw provider value/source-unit metadata. Forecast/errors/quantiles use cumulative log-return units; arithmetic return=exp(logreturn)-1. Saved forecast prices refer to origin (INRIndia/USDUS), not refreshed quotes. Probability is0..1 only when frozen accepted calibration exists; null means no issued calibrated probability. raw_probability is a separate diagnostic score. Missing/nonfinite values and unsupported relative errors remain null. RawClose, total-return AdjClose and corporate actions are separate. Nifty benchmark is price index, notTRI.

## Research semantics/privacy
ForecastResponse explicitly provides RESEARCH_ONLY, promotionBLOCKED, recommendation_allowed=false, origin/group/protocol/model/calendar identities, cutoff/price timestamp and horizons. Every horizon remains LOW/abstain with reasons, OOD, regime, data status, training cutoff, feature schema and opaque source hashes. Clients must never infer BUY/SELL from raw scores or confidence.

No route serves private raw snapshots, feature vectors, model files or paths. Explanations are saved drivers. Untouched-final and boundary-crossing predictive outcomes are masked; counts remain operationally visible. Enough evidence means eligible for review, never promotion.

## Caching/performance
Frozen metadata are stable. Native collection models cache by frozen identity after artifact/schema validation. Successful provider histories use a30-second in-process cache keyed by provider/symbol/5y/1d/reference role. Provider failures are not cached as successes; no stale-on-error substitution exists. Saved forecasts/validation read explicit ledger origins/versions and are not cached indefinitely.

Instrumentation records median/p95 for normalized routes and provider.history; optional private JSONL persists samples. /system/performance returns aggregates. Saved forecast retrieval performs no inference; its inference latency is not fabricated as0. Collection inference timings are separate. Small samples/cache/startup measurements are not capacity guarantees. Indexing/aggregation optimization follows actual measurements.

Forecast history exposes typed saved predictions and separately realized/pending/sealed outcomes. Legacy price timestamps are derived from the actual origin session, rather than the newest observed provisional bar. Hash-bound timestamp amendments identify later audited acceptance; original outcome payloads and return values remain preserved. Provider-cache capacity is128keys with expired entries evicted.
