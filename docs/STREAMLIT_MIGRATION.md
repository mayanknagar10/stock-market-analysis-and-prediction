# Streamlit migration

Next.js is the new product workspace. Streamlit remains available as the internal/research fallback and is not deleted. The migration adds Python application-service adapters and FastAPI routes around existing financial/forecast exports; calculations are not rewritten in React. Existing Streamlit pages continue using the same Python core/research functions.

| Existing area | New workspace |
|---|---|
| Dashboard / market context | / and /markets |
| Overview / technicals / fundamentals / factors | /stocks/[ticker] sections |
| Saved forecasts / scenarios / explanations | /forecast/[ticker] tabs |
| Historical validation / forward evidence | /validate |
| Risk report | /risk/[ticker] |
| Portfolio analytics | /portfolio |
| Watchlist tracking | /watchlist |
| Bounded screening | /screener |
| Multi-stock comparison | /compare |
| Manual collection / outcomes / system status | /admin/research-status |
| Strategy backtester | Local-operator internal blueprint only; execution remains in Python / Streamlit |

## Product presentation

The web workspace uses a dense terminal composition instead of reproducing Streamlit layouts: compact command/search chrome, persistent instrument context, large financial charts, adjoining metric strips and result tables. Simple/Advanced disclosure keeps diagnostics accessible without making every detail the default. Dashboard, Stock, Forecast, Validate and Markets have distinct analytical layouts; Compare, Screener and Portfolio use input/result workbenches. Dark and light themes share semantic tokens. See ../frontend/DESIGN.md for the implemented specification.

## Retained differences

Streamlit account identity and local saved user files are not migrated into web authentication. Browser guest watchlists/portfolio inputs are separate, with authenticated stateless Python calculations. Existing guest records are not silently imported. Hosted accounts and ownership remain a controlled later step.

Backtesting, older strategy/admin utilities, personal notifications and assistant workflows remain in the research console where they lack an approved web API boundary. The local-operator backtesting blueprint exposes input structure and disabled result panels; it cannot execute a strategy and is excluded from ordinary navigation. Other unsupported workflows are not replaced with fake UI pages. The web screener processes an explicit bounded list, not an automatic100-provider scan. Portfolio values describe hypothetical constant-weight analysis, not broker accounts or execution.

The first cloud scheduler/storage request was changed by the user to local PC/backend plus protected manual buttons. Laptop independence remains deferred; Streamlit is no longer needed to invoke the standalone jobs. Opening any ordinary web page remains read-only and never creates a forecast.

## Research truth

Failed V5 results and the consumed final remain permanently visible in Validate. Canonical forward collection, preserved legacy/manual records and retrospective reconstructed results stay separate. The preservation baseline is 388 legacy saved forecasts (97 matured, 291 pending) and zero canonical forecasts; displayed counts come from the selected backend evidence stream. Models, thresholds and abstention remain frozen; production stays blocked. No successor is trained by this migration.

## Verification and retirement

The fresh backend suite includes retained Streamlit initialization/smoke tests. Final frontend/browser/build results are listed in WEB_PLATFORM_COMPLETION_REPORT.md. Do not remove the fallback until hosted authentication, data migration, all remaining workflow requirements and production deployment checks are independently approved and verified.
