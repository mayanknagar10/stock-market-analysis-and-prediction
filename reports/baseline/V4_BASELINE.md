# V4 baseline evidence

Date: 07 October 2026. This is a **retrospective diagnostic**, not an out-of-sample performance claim. Frozen checkpoint was trained on synthetic tickers; original training dates/data are unavailable. Full machine-readable metrics and ticker/sector/family/year/volatility/regime breakdowns: V4_DIAGNOSTIC.json. Every scored observation, deterministic simulation seed and method is in V4_DIAGNOSTIC.csv.

Dataset: six equities (RELIANCE.NS, TCS.NS, HDFCBANK.NS, AAPL, JPM, XOM), captured once with raw OHLCV, Adj Close, splits/dividends and content hashes. 12 development-period origins per ticker, 72 observations per horizon, 30 simulated paths. No origin/label endpoint reaches final-test start 2025-10-01. Provider snapshot: data/snapshots/10775f12a4c4de8c1e3285168a278d14828e197cd41a66cf909220919f555690.json. Local snapshots are excluded from Git; archive them privately with their manifest for reproduction.

| Horizon | V4 recursive MAE (log return) | RMSE | Direction hit rate | Simulated 80% coverage | Zero-return MAE |
|---|---:|---:|---:|---:|---:|
| 1D | 0.010502 | 0.014181 | 45.83% | 86.11% | 0.009849 |
| 5D | 0.037112 | 0.058502 | 48.61% | 75.00% | 0.033836 |
| 10D | 0.037031 | 0.049723 | 62.50% | 72.22% | 0.037248 |
| 20D | 0.048550 | 0.061995 | 52.78% | 76.39% | 0.047633 |

The 10D sign result does not establish an edge: the sample is sparse, labels overlap and the original checkpoint has no verifiable real-market training/OOS lineage. V4 performs worse than zero return on most return errors in this diagnostic. Report all results, including unfavorable ones. Currency-denominated price errors pool INR/USD and cannot be interpreted as a meaningful cross-market score; use family/ticker breakdowns.

The report separately evaluates the actual recursive simulation and the README's one-day compounding formula. Neither is silently substituted for the other. Historical-drift baseline is included. Brier, log loss, probability calibration and model-quantile intervals are unavailable because V4 does not output them. The 90% interval is unavailable (q10/q90 is 80%). These values remain null. A forecast ledger/trading policy does not exist in V4, so forecast-strategy Sharpe/Sortino/profit factor/turnover are not manufactured. Existing vectorbt backtests are separate research outputs, with their own execution/cost limitations. Market/sector beta comparison needs benchmark/sector snapshots and is an outstanding baseline extension.

## Reproduction

Use the frozen code/artifact hashes in V4_FREEZE.json. Install requirements plus requirements-dev.txt in an isolated environment. Then run:

~~~powershell
python scripts/capture_baseline_data.py
python scripts/evaluate_v4.py --manifest PATH_TO_CAPTURED_MANIFEST --origins 12 --normalize-v4-text --output reports/baseline/NEW_DIAGNOSTIC.json
python -m pytest -q
~~~

Use the same original manifest rather than recapturing prices to reproduce the same predictions. capture_baseline_data is an explicit network job; evaluation itself is offline. Reports use exclusive-create writes; choose a new output for another run. Seed is derived from snapshot SHA and origin date; this removes Python hash randomization without changing numeric tree parameters.

## Platform finding

The Windows checkout uses CRLF in universal_lgb.txt, but the booster tree_sizes offsets were computed for LF. Native file loading emits model-format errors. The diagnostic adapter verifies all frozen artifacts/source hashes and explicitly normalizes an in-memory copy to LF. Numeric trees and original bytes remain unchanged. This is recorded in JSON (v4_text_normalized=true). The current production loader still needs the later hardening fix; the smoke check of prediction landing explicitly uses the normalized diagnostic loader, so it is not evidence that unmodified native loading is healthy.

## Current checks

14 metric/preservation/loader tests passed; 13 existing-page initialization tests passed with mocked network providers (including missing optional global/news feeds). Initial states, not every interactive workflow, are covered. Eight existing UTC-naive datetime deprecation warnings remain. No lint/type config existed. A real server startup check is recorded in the implementation status.
