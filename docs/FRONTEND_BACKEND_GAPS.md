# Frontend / backend capability roadmap

This phase completes the product interface using current APIs. Missing data remain unavailable, hidden from ordinary navigation or restricted to clearly labeled internal previews. This roadmap does not authorize provider expansion, model research, cloud work or deployment. Python remains authoritative for future calculations.

Priority: P1 = next backend integration needed for an existing workspace; P2 = deeper supported-product analysis; P3 = optional future specialty. Proposed routes below are designs only, not implemented endpoints.

| Page / component | Desired data | Existing endpoint | Missing field / capability | Proposed future endpoint | Priority |
|---|---|---|---|---|---|
| Backtesting internal blueprint | Existing strategy execution, fees/slippage, equity/drawdown, trade log and metrics | None; strategy functions remain in Python / Streamlit | Authenticated bounded asynchronous strategy job contract | POST /api/v1/analysis/backtests; GET /api/v1/analysis/backtests/{id} | P1 |
| Dashboard / Markets Sensex | Sensex current daily session context | GET /markets/context | No configured Sensex reference in current19-source service | Extend market context via separately approved UI data source | P1 |
| Global search | Broader market/company universe | GET /stocks/search | Frozen100NSE metadata only; no fullUS or BSE company directory | Extend /stocks/search with registered market filter/metadata | P1 |
| Unknown ticker classification | Distinguish a non-listed symbol from a temporary provider outage | GET /stocks/{ticker}/overview | A syntactically valid unknown symbol such as `ZZZZZZ.NS` currently returns `503 PROVIDER_FAILURE`; the frontend can only show the backend's provider-failure state | Resolve registered/known symbol status before provider lookup and return a distinct structured code | P2 |
| Stock bar freshness | Compare a stock's completed bar with its expected exchange session | GET /stocks/{ticker}/overview; GET /data/health | The response exposes a price timestamp and `SESSION_BAR_NOT_LIVE_QUOTE`, but no per-ticker calendar-relative stale/current verdict; the frontend displays the exact timestamp without inventing one | Add backend-owned per-ticker freshness classification | P1 |
| Stock executive summary | Period returns, daily move, concise technical/risk states | GET /stocks/{ticker}/overview, /technicals/series, /risk/report | No combined typed summary or standard multi-period return fields | GET /stocks/{ticker}/research-summary | P1 |
| Dashboard Market Pulse | Registered trend, breadth and regime state | GET /markets/context | Daily prices/change only; no breadth/trend summary | GET /markets/pulse | P2 |
| Sector view | Full industry/sector coverage and constituent breadth | GET /markets/context, /markets/sectors | Limited registered index proxies;17industries are not17official sector indices | GET /markets/sector-performance | P2 |
| Portfolio account snapshot | Actual positions/P&L/daily move, sector exposure/correlation | POST /analysis/portfolio; /analysis/watchlist | Portfolio endpoint is hypothetical daily constant-weight analysis; no real account ledger | Separate authenticated portfolio account/analysis contracts | P1 |
| Watchlist compact sparklines / alerts | Price-series summary and persisted alert state | POST /analysis/watchlist; GET /stocks/{ticker}/history/series | Watchlist has no sparkline or alert event fields; target/stop booleans are tracking only | Extend watchlist analytics; later authenticated alert API | P2 |
| Screener volatility / factor / cap filters | Declared-unit filters and compact measured columns | POST /analysis/screen | Available filters: RSI, P/E, beta, dividend, EMA50 trend; no volatility/cap/factor request fields | Extend bounded ScreenRequest using existing Python calculations | P2 |
| Compare | Matched fundamentals/technicals/factors and multi-stock drawdowns | POST /analysis/compare plus individual report GETs | Comparison contract principally normalized performance/correlation/risk; separate reports have different availability/vintages | GET/POST typed comparison-detail adapter | P2 |
| Risk return distribution / monthly heatmap | Backend-computed distribution/monthly returns | GET /stocks/{ticker}/risk/report | Metrics/drawdown available; no standardized distribution/monthly heatmap series | Extend RiskResponse with typed series | P2 |
| Fundamentals history | Growth, cash-flow, earnings and valuation history | GET /stocks/{ticker}/fundamentals | Current revised snapshot; no PIT historical financial series | GET /stocks/{ticker}/fundamentals/history | P2 |
| News/events | Market news, filings, scheduled corporate events | GET /stocks/{ticker}/news; /history/series | Company news/VADER and historical dividend/split fields only; no filings/event calendar/market-news feed | /markets/news; /stocks/{ticker}/events after source approval | P2 |
| Forecast model agreement | Validated agreement/disagreement measurement | GET /research/forecasts/{id} | Component scores exist; no validated agreement score | Extend archived detail with versioned agreement semantics | P2 |
| Forecast invalidation | Backend-defined conditions with research provenance | Existing scenario POST | No validated invalidation rules | Add versioned invalidation output only after separate research authorization | P3 |
| Analogues | Similarity score and richer neighbour metadata | GET /research/forecasts/{id}/analogues | Matured outcomes/counts available; no issued similarity score | Extend typed analogue metadata | P2 |
| Validate V4/V5 comparison | Preserved original comparative report | GET /validation/historical/v5/{report} | No dedicated V4-vs-V5 comparison contract | Read-only pinned historical comparison endpoint | P2 |
| Model/Data Health | Consistent training cutoff, per-source freshness, latest successful batch | Model/status, collection/status and forecast metadata | Some fields available only per-prediction/batch, no combined typed health contract | GET /research/health-summary | P2 |
| Hosted accounts | Secure identities, owned guest-data migration and revocation | Local operator session / bearer | No full hosted account authentication | Dedicated controlled auth migration | P1 before hosted sensitive workflows |
| Laptop-independent collection | Approved scheduler and private durable storage | Existing CLI/local scheduler | PC awake/logged-in dependency; cloud explicitly deferred | Separate approved deployment/storage plan | Deferred |

## Specialty modules remain future work

Options, options flow, open interest, IV/GEX, futures, FII/DII, institutional activity, bulk/block deals, advanced alerts and geopolitical intelligence have no approved contracts/data pipeline in this phase. They are documented here rather than exposed as fake ordinary product pages. Advanced NLP is not implied by VADER.

## Non-negotiable integration semantics

No API gap justifies fabricated prices, performance, probabilities, forecasts or recommendations. Canonical and legacy/manual prospective evidence never pool with reconstructed history. Sealed/pending outcomes remain unavailable. Model/threshold/abstention/final results are unchanged and production stays blocked. An internal backtesting preview is a product blueprint, not an executed strategy result or ML validation claim.

The controlled backtesting form also needs a Python-owned strategy catalogue (names, parameter bounds/defaults and supported cost conventions) before execution is enabled. The current internal preview mirrors existing strategy metadata; it performs no strategy calculation. Proposed GET /api/v1/backtesting/strategies is not implemented.

## OpenAPI surface audit (2026-10-10)

The checked-in OpenAPI schema defines 44 operations. The typed frontend client uses 32, mapped as follows. The backend remains the source of all financial calculations.

| Frontend surface | Connected operations |
|---|---|
| Shell, Dashboard, Validate health, internal Research status | `GET /health`, `/model/status`, `/data/health`, `/collection/status`, `/system/status`; `GET`, `POST`, `DELETE /operations/session`; `GET /operations/status`, `/operations/jobs`, `/operations/jobs/{run_id}`; `POST /operations/jobs` |
| Global search and Stock workspace | `GET /stocks/search`, `/stocks/{ticker}/overview`, `/fundamentals`, `/news`, `/history/series`, `/technicals/series`, `/risk/report`, `/factors/report` |
| Markets and Dashboard references | `GET /markets/context` |
| Forecast and prediction history | `GET /stocks/{ticker}/forecast`, `/research/forecasts`, `/research/forecasts/{prediction_id}`, `/research/forecasts/{prediction_id}/analogues`; `POST /research/forecasts/{prediction_id}/scenario` |
| Validate | `GET /validation/prospective`, `/validation/historical/v5/{report}` |
| Portfolio, Watchlist, Compare, Screener | `POST /analysis/portfolio`, `/analysis/watchlist`, `/analysis/compare`, `/analysis/screen` |

All paths above are under `/api/v1`. The remaining 12 are accounted for below; none requires a fabricated frontend result.

| Unused operation(s) | Current frontend path or disposition |
|---|---|
| `GET /markets/overview`, `/markets/sectors` | Richer `GET /markets/context` supplies the Market and Dashboard views. |
| `GET /model/health`, `/validation/summary` | `GET /system/status`, `/model/status`, and the explicit canonical/legacy `/validation/prospective` streams supply these views. |
| `GET /screener` | Frozen ticker metadata comes from `/stocks/search`; selected-universe screens use `POST /analysis/screen`. |
| `GET /stocks/{ticker}/factors`, `/risk`, `/history`, `/technicals` | Typed `/factors/report`, `/risk/report`, `/history/series`, `/technicals/series` serve the Stock workspace. |
| `GET /stocks/{ticker}/forecast/explanation`, `/stocks/{ticker}/forecast/history` | Saved prediction detail and immutable history use `/research/forecasts/{prediction_id}` and `/research/forecasts`. |
| `GET /system/performance` | Internal instrumentation is inspected in QA; no normal-product view is needed. |
