# Frontend architecture

## Application boundary

The dedicated frontend/ project uses Next.js App Router, React and strict TypeScript. It renders shared Python/FastAPI service responses; it does not implement forecasting, indicators, portfolio returns, calibration, OOD, SHAP or validation calculations in JavaScript. Streamlit remains the internal research fallback.

The authoritative schema is docs/api/openapi.json. npm run generate:api writes frontend/src/lib/generated/api.d.ts using the pinned openapi-typescript generator. DTOs refer to generated components; request paths use generated paths via openapi-fetch. application/web_schemas.py supplies additive typed contracts while the original API responses remain compatible.

## Directories

| Directory | Purpose |
|---|---|
| src/app | URL-addressable route wrappers, layout and semantic theme tokens |
| src/features | Dashboard, Forecast, Validate, stock research and analytical workflows |
| src/components | Shared shell/search, query states, status, tables, charts and providers |
| src/lib | Single API client, query keys, formatting, browser guest data and generated API types |
| src/test | Unit-test setup and fixture/mocking support |
| e2e | Isolated headless browser tests and backend-derived contract fixtures |

## Data and state

The API client normalizes NEXT_PUBLIC_API_BASE_URL, validates HTTP(S) configuration and keeps fetch calls in one layer. Missing configuration produces a visible error. Browser credentials use include for explicit local operator cookies. A manually entered private bearer remains in memory; no credentials are persisted as guest data.

TanStack Query deduplicates requests by resource/ticker/evidence-group/range keys. Provider current-vintage data use bounded freshness; saved forecasts and validation are versioned by evidence group and origin-sensitive response metadata. Only active tab explanations/analogues are fetched. Page reads never train or capture a forecast.

Browser-local preferences include theme, Simple/Advanced disclosure mode, recent selected companies and guest positions. These are distinct from the retained Streamlit account store. Sensitive analysis POSTs require operator access; Python returns all financial results. Forms validate bounded inputs and show backend errors without raw tracebacks.

## Product routes

Dashboard /; company /stocks/[ticker] with overview, chart, forecast, technicals, fundamentals, news, factors, risk and prediction-history sections; /forecast/[ticker]; /validate; /markets; /risk/[ticker]; /portfolio; /watchlist; /screener; /compare; /admin/research-status. Query parameters preserve evidence group and workspace tab.

The canonical prospective group defaults to an honest empty state when no forecasts exist. Preserved manual forecasts require an explicit legacy selection. Reconstructed historical V5 results are a separate stream; the failed consumed gate remains visible. LOW/abstained research estimates never become BUY/SELL recommendations. Current session prices remain distinct from immutable origin prices.

## Charts and interaction

Lightweight Charts provides candlesticks, volume, crosshair values and time ranges. Dates use exchange-session labels supplied by Python, not browser UTC date guessing. Recharts provides lazy analytical views of already computed series and metrics. Text summaries/table alternatives supplement charts.

Shared skeleton, unavailable, degraded, pending, sealed and error states preserve layout and provenance. Desktop navigation becomes a focus-contained mobile drawer; search supports debounce, arrows, Enter, Escape and recent selections. Dark/light themes share CSS variables. The terminal shell uses a 176px sidebar, compact command/context bars and a chart-led main workspace. Dashboard and stock views pair charts with contextual readouts; Validate separates a frozen-decision rail from forward evidence; Compare, Screener and Portfolio pair bounded input rails with result panels. InstrumentHeader shares company/quote/timestamp/data-health presentation. WorkspaceMode persists Simple/Advanced disclosure and AdvancedDetails opens diagnostics accordingly. See frontend/DESIGN.md and UI_DESIGN_SYSTEM.md for actual tokens, breakpoints and verification limits.

The schema currently exposes 44 HTTP operations across 41 paths. The operator-only backtesting preview has disabled execution because no web strategy-job API exists; it is not an analysis endpoint.

## Manual research operations

Admin / Research Status authenticates separately before capture, finalize, outcome resolution or preservation verification. It submits fixed job types, polls server records and displays failures/degradation/skips honestly. The unchanged CLI calendar/cutoff guards own eligibility. This approved mode requires a running local PC/backend; cloud independence remains deferred.

## Verification and deployment

Unit/contract tests, strict type checking, lint, production build and isolated browser QA are recorded in WEB_PLATFORM_COMPLETION_REPORT.md. Browser fixture data are never served by production routes. The Next.js frontend requires no writable deployment filesystem; Python storage/scheduling/auth assumptions remain separate. See FRONTEND_DEPLOYMENT.md, BACKEND_DEPLOYMENT.md and AUTH_MIGRATION.md.
