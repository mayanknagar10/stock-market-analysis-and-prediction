# StockPro Analytics

StockPro is a research-first financial analytics and forecasting platform focused on transparent evidence, market context and multi-horizon forecasts. Its primary web application is Next.js with a Python/FastAPI backend. Streamlit remains an internal research console.

> **RESEARCH ONLY.** Production model promotion is **BLOCKED**. The frozen V5 candidate `research-20261008T190859` did not pass its final gate, and no validated statistical edge has been certified. Forecasts may have LOW confidence and abstain. Prospective validation is still accumulating; these outputs are not trading recommendations.

## Architecture

`Browser → Next.js / React / TypeScript → FastAPI → Python analytics, research services and data providers → preserved models, ledgers and source snapshots`

The frontend uses Tailwind CSS, TanStack Query, generated OpenAPI types, TradingView Lightweight Charts for price charts and Recharts for analytical views. TradingView provides chart rendering here; StockPro does **not** scrape TradingView market data. Python owns forecasting, technicals, factors, risk and validation calculations. The research stack includes XGBoost and LightGBM; normal API requests do not train models.

The versioned API is under `/api/v1`. Its contract is in [OpenAPI](docs/api/openapi.json) and [API documentation](docs/API_CONTRACT.md). The Next.js app consumes that contract through one typed client. The Streamlit app remains available for internal research and administration.

## Product areas

| Area | Current scope |
| --- | --- |
| Dashboard and Markets | Latest available Indian/global reference sessions, watchlist context and research/collection health. Daily bars are not live quotes. |
| Stock Research | Price history, technicals, available fundamentals, factors, risk and provider news. Missing provider data is shown as unavailable. |
| Forecast | Saved 1-, 5-, 10- and 20-session research forecasts, uncertainty, model drivers where archived, conditional scenarios, historical analogues and immutable prediction history. |
| Validate | Failed V5 gate, retrospective research and separate prospective predictions/outcomes with sample counts and abstention. |
| Screener and Compare | Backend-calculated screening and stock comparisons; local operator access is required for protected calculations. |
| Portfolio and Watchlist | Browser-local tracking inputs with protected backend valuation/risk calculations; no broker execution. |
| Backtesting | Internal research blueprint and existing Python/Streamlit tools; a general authenticated web backtest job is not yet available. |
| News / Events | Provider-supported news and event context when available; empty and unavailable states are explicit. |
| Internal research/admin | Operator-only manual jobs and research status, separate from ordinary product navigation. |

Main web routes include `/`, `/markets`, `/stocks/[ticker]`, `/forecast/[ticker]`, `/validate`, `/screener`, `/compare`, `/portfolio`, `/watchlist`, `/risk/[ticker]` and `/admin/*`. Stock factors and news live inside the stock workspace.

## Forecast and evidence policy

The frozen V5 research architecture predicts each horizon directly rather than compounding a one-session prediction. It combines return regression, direction estimates and quantile/uncertainty outputs where saved. Calibration status, out-of-distribution checks, data quality and abstention are exposed explicitly. An unverified probability is not presented as calibrated confidence or a BUY/SELL recommendation.

Research uses temporal splits and leakage controls. The consumed V5 final period remains visible and cannot be reused for tuning. Historical reconstructed evaluation is labeled separately from genuinely prospective forecasts recorded before outcomes. Saved predictions are immutable; matured outcomes are attached as separate records. Promotion requires a future pre-registered gate and sufficient prospective evidence, not elapsed time alone.

Market data currently come primarily from Yahoo Finance through `yfinance`; supported research/legacy paths also use Stooq as a best-effort historical fallback, SEC EDGAR filings, Kenneth French factor data and VADER-based sentiment on available headlines. Free providers can be delayed, incomplete or unavailable. Provider timestamps and data-health states govern what the product shows.

## Local development

Python 3.12 is the verified development version. From the repository root, create an environment and install backend and test dependencies:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements-dev.txt
```

Set `STOCKPRO_CORS_ORIGINS` to `http://127.0.0.1:3000` in the backend process environment, then start FastAPI:

```sh
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

The API health endpoint is `http://127.0.0.1:8001/api/v1/health`; its interactive schema is at `http://127.0.0.1:8001/docs`. In a second process, start the web app:

```sh
cd frontend
npm ci
# Copy .env.example to .env.local and set NEXT_PUBLIC_API_BASE_URL.
npm run dev
```

Open `http://127.0.0.1:3000`. The example URL targets the local backend; configure an actual hosted backend URL for deployment. The root [backend environment example](.env.example) lists optional settings, and [frontend/.env.example](frontend/.env.example) shows the public API URL. Example files are documentation; set backend variables in the process environment or your secret manager. Never commit actual credentials or private raw snapshots. Manual operator jobs are disabled by default and require the explicit local loopback configuration described in [scheduling](docs/SCHEDULING.md). Streamlit can be started separately with `python -m streamlit run app.py` for internal use.

## Verification

From the repository root:

```sh
python -m pytest -q
python scripts/verify_automation_preservation.py
```

From `frontend/`:

```sh
npm run typecheck
npm run lint
npm test
npx playwright install chromium
npm run test:e2e
npm run build
```

Playwright tests need a running local frontend. The fixture-based browser suite does not replace checks against a real FastAPI instance. The preservation command verifies frozen V5 and prospective artifacts without regenerating research. See [frontend/backend gaps](docs/FRONTEND_BACKEND_GAPS.md) for capabilities that remain unavailable.

## Deployment status and limitations

The Next.js frontend is designed for a Vercel-compatible build; the Python API is an independently hosted service. **Neither has been deployed as part of this repository state.** Hosted authentication, persistent private backend storage and approved unattended scheduling are prerequisites before relying on remote collection. The current approved collection mode is local/manual and requires the local backend to be running. Provider latency can delay individual sections; some frontend/backend capability gaps remain; prospective evidence is insufficient for production.

StockPro is for research and education. It does not guarantee price accuracy or profitability and does not place trades.
