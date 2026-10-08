# StockPro Analytics V5

StockPro is a Python/Streamlit equity research application with preserved technical, risk, portfolio, strategy, factor, screener and market tools. V5 adds direct 1/5/10/20-session forecasts, conditional distributions and an auditable validation workspace.

**Release state: RESEARCH ONLY. The frozen performance gate failed. Production inference is disabled.** Historical availability/revision archives are absent, so reconstructed contextual inputs cannot be certified point-in-time. All research forecasts remain LOW confidence and abstain. The evidence does not support relying on this candidate for trading decisions.

## Start locally

Python 3.12 was used for verification. Create an isolated environment:

~~~powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m streamlit run app.py --global.developmentMode=false
~~~

Open http://localhost:8501. The development session used an ignored .deps directory; it is not required by the application. See [deployment instructions](docs/V5_DEPLOYMENT.md) for explicit capture/training/outcome jobs and durable private storage.

Default V5 feeds are free/public. Prediction does not require an external LLM or paid API key. Free feeds may be stale, incomplete or unavailable; critical source failures prevent a forecast. SEC requests require an actual owner-supplied STOCKPRO_SEC_USER_AGENT contact identity.

## Navigation

| Workspace | Contents |
|---|---|
| Dashboard | Market context, source status and the master task board |
| Research | Overview, technical analysis, compare, screener, factors and insights |
| Forecast | Selected-horizon return/price, 50%/80% ranges, calibration state, source/OOD warnings, SHAP associations, scenarios and analogues |
| Validate | Development/final metrics, calibration, model cards, quarantines and immutable prediction history |
| Risk & Portfolio | Risk analysis, portfolio and watchlist |
| Markets | Market overview and global context |

Existing analytics remain available. The assistant forecast uses the same V5 research service. Normal page loads never train models. V4 artifacts and diagnostic adapters are retained for comparison; the V5 multi-session path uses independent direct heads.

## Model and data architecture

India and US equity families have separate XGBoost/LightGBM regression, direction and native quantile heads for each horizon. Returns are predicted directly, then converted to prices. Models use causal scale-free technical features plus development-tested context groups. Four disjoint purged blocks separate fitting, blend weights, calibration and development validation. Rejected probability calibrations are shown as raw scores.

Sources flow through core/data, with dated immutable snapshots, provider/adjustment semantics and explicit health flags. Corrupt snapshots, incompatible schema/family, missing critical prices and unknown session bars fail closed. Independent observed session calendars are research proxies, not verified exchange calendars. Predictions and matured outcomes are append-only SQLite records with integrity hashes and complete source/model lineage.

## Measured result

Candidate: research-20261008T190859. Final test starts 2025-10-01 UTC and uses only outcomes matured by the original 2026-10-07 capture cutoff. The period was locked and consumed once; it must not be reused for tuning.

| Frozen requirement | Observed | Required |
|---|---:|---:|
| MAE better than zero-return | 1/8 groups | 6/8 |
| RMSE better than zero-return | 0/8 | 6/8 |
| Brier below 0.25 | 2/8 | 6/8 |
| 80% interval coverage between 70% and 90% | 7/8 | 6/8 |
| Development-accepted calibrated heads | 3/8 | 8/8 |

[Full V4 versus V5 report](reports/V4_VS_V5.md) includes matched recursive legacy origins, naive baselines, direction/calibration, sector/regime/year/stock breakdowns and auxiliary excess-return metrics. The shipped V4 checkpoint was trained on synthetic tickers and cannot establish real-market historical OOS accuracy.

Verification: **122 automated tests passed**, with eight retained legacy UTC deprecation warnings. Tests cover temporal contracts, purging, snapshots/providers, models, probabilities/quantiles, native registry integrity, scenarios, immutable ledger/outcomes and existing/new page initialization. Browser visual/accessibility QA was unavailable and is not claimed as passed.

## Status and limits

[Master task board](docs/V5_TASK_BOARD.md) shows complete, deferred and blocked tasks. [Completion report](docs/V5_COMPLETION_REPORT.md) records the delivered scope and remaining work. [Architecture](docs/V5_ARCHITECTURE.md), [source inventory](docs/DATA_SOURCES.md) and [phase record](docs/V5_IMPLEMENTATION_STATUS.md) explain the contracts and evidence.

The current research universe is RELIANCE.NS, TCS.NS, HDFCBANK.NS, AAPL, JPM and XOM, with limited sector mappings. Other tickers are not validated by this experiment and may fail required source/schema checks. Crypto/index/FX forecast families and 60-session forecasts are unsupported. News/NLP expansion remains performance gated; existing VADER is preserved, and event provider interfaces fail safely without claiming coverage. No historical production track record is fabricated.

Preserve private snapshots, users and ledger on durable storage; they are excluded from Git. Do not promote this checkpoint or delete its failed/quarantined history. Future model work requires a new untouched holdout and stronger data lineage. No forecast guarantees profit or future price accuracy.
