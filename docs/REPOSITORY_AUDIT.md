# StockPro V4 repository audit

Audit date: 07 October 2026. Frozen commit: b173a33b1a61547077d6ead3a2746c3eeef981a7. Branch: v5-development. The checkout was clean before work. No AGENTS.md or existing automated tests/lint configuration were found. All original tracked files are inventoried with SHA-256 in reports/baseline/V4_FREEZE.json; the V4 files and checkpoint bytes remain unchanged.

## Application and preserved modules

| Responsibility | Verified implementation | Findings / migration treatment |
|---|---|---|
| Entry/navigation | app.py; st.navigation/st.Page | 13 active pages in 5 groups. About says v6 although model says V4. Eight garbled-filename page copies are inactive under explicit navigation. Preserve until smoke tests cover replacements. |
| Research/dashboard | pages/overview.py, technical_analysis.py, compare.py, screener.py, factor_analysis.py, insights.py | All use shared core fetch helpers; global_data uses external_data directly. Preserve analytical functions. |
| Price prediction | pages/price_prediction.py; core/models.py | Explicit retrain panel, but missing checkpoint also triggers fitting on ordinary forecast requests. Full-history checkpoint is used by rolling historical inference. |
| Technical features | core/indicators.py | 56 scale-free technical/calendar columns; batch implementation duplicates formulas for simulated paths. No contextual features. Trailing rolling/ewm computations appear causal; automated prefix/scale tests required. |
| Risk | core/risk_metrics.py; pages/risk_analysis.py | Local VaR/CVaR, drawdown, CAPM, GBM Monte Carlo; fixed risk-free rate/252-session assumption. Preserve calculations; reference simulations are not model uncertainty. |
| Strategy backtester | core/strategy_backtest.py; pages/backtester.py | vectorbt strategies, explicit fee/slippage; same-window grid optimization has data-snooping risk. Signals computed at close execute at close; economic executability needs separate review. |
| Screen backtest | core/screen_backtest.py; pages/screener.py | Excludes present-day fundamental filters, a useful existing protection. Current universe induces survivorship bias; equal-weight basket ignores costs. Intersection of ticker calendars drops missing observations. |
| Factors | core/factor_models.py; pages/factor_analysis.py | Fama-French monthly OLS and local quant ranking. Fetched factor vintages/publication times are not persisted. |
| Portfolio/watchlist | pages/portfolio.py, watchlist.py | Streamlit session-state positions, examples/defaults, alerts; no durable portfolio transaction store. Preserve functionality and user state. |
| Auth/personalization | core/auth.py, personalization.py | Salted PBKDF2; local data/users.json; import seeds demo user; separate JSON read/modify/write can lose concurrent updates. Never import during audit against user storage. Back up privately before any migration. |
| Notifications | core/notifications.py | In-session inbox and browser Notification API; remote icon. SMTP/Resend code is inside a documentation string, not a live external integration. No messages sent in this audit. |
| Sentiment | core/sentiment.py | Offline VADER + finance lexicon; simple averages, flat legacy Yahoo headline schema, UTC-naive date grouping. Missing VADER may produce neutral results; neutral must not imply actual analysis. |
| Filings | core/external_data.py; pages/insights.py, global_data.py | SEC CIK/submissions metadata and document links; no automated full-document/XBRL fetch. Filing descriptions/manual pasted excerpts are not full filing analysis. Placeholder SEC contact needs owner configuration. |
| Assistant | core/assistant.py | Rule-based analytics explanations; forecast handler calls V4 forecast_future directly, bypassing any future V5 UI gate. Must migrate when prediction path changes. |
| Global/macro | core/external_data.py; pages/global_data.py, market_overview.py | CoinGecko, Frankfurter, World Bank and Yahoo indices/commodities. Forecast uses none of these as inputs today. |
| Caching | core/cache_layer.py, st.cache_data | Yahoo 300s, fundamentals 3600s disk cache plus Streamlit TTL; other feeds Streamlit cache. No source vintages/snapshots; exceptions mostly hidden. Model singleton does not invalidate automatically if files change. |
| Checkpoints | models/universal_xgb.json, universal_lgb.txt, universal_scaler.json, universal_meta.json | Native boosters and JSON robust-scaler arrays. No training date boundaries, target/schema hash, code/source versions or integrity hashes. Partial model loading accepted; missing features zero-filled. |
| Legacy artifacts | lit.ipynb, keras_model.h5 | Notebook downloads Yahoo TATAMOTORS and trains LSTM with test data as early-stopping validation. Not imported by active app; do not introduce it into V5. CSVs are ticker/period lists, not OHLCV history. Thesis PDF is a legacy research artifact, not executable architecture. |
| Presentation | utils/helpers.py, charts.py; .streamlit/config.toml | IBM Plex remote fonts, gradients/signal badges, table formatting and Plotly layout helpers. Reuse computation/escaping; redesign only after modeling gate. |

## Prediction and training details verified from code

1. _build_training_row_set creates tomorrow's log return from Close.shift(-1). Missing/infinite feature rows are dropped for training. Latest-row inference forward-fills and zero-fills, including unknown schema columns.
2. train_universal_model pools 20 India + 20 US equities. It splits each ticker independently at 85%, discards calendar dates in pooled frames, fits RobustScaler on train rows, trains 600-tree XGB/LGB regressors, and blends equally. Calendar cutoffs differ by ticker; future labels at split boundaries are not purged. No probabilities, quantiles or true calibration exist.
3. The actual current forecast_future uses recursive Monte Carlo, not the README's simple compounding formula: each synthetic daily OHLCV trajectory rebuilds technical features and queries the one-day model, then adds historical-volatility Gaussian noise. The endpoint median and 10/90 percentiles are simulation outputs. The optional seed uses Python hash(), which changes across process runs. _vol_cone remains separately in the module. Both V4 recursive inference and README compounding must be distinguished in comparison reports.
4. walk_forward_backtest ignores its horizon argument when building labels: every evaluation is one-day. If a checkpoint loads, it never retrains per fold or verifies training cutoff. Those results cannot be called OOS. The fallback trains per fold without a label gap and reports price errors plus sign accuracy.
5. The shipped metadata says the training universe was SYN_0 through SYN_14. Saved MAE 0.016877, RMSE 0.021685, directional accuracy 48.86% are synthetic checkpoint metadata, not fresh real-equity evidence. Raw training data and generation seeds are absent; exact training reproduction is unavailable. Preserve these numbers unchanged and label them.

## Critical release risks

- Present checkpoint is synthetic and mixes asset-family assumptions; no defensible live-equity accuracy record.
- Historic inference may use a checkpoint trained later; per-ticker pooled split allows cross-stock calendar leakage and overlapping label endpoints.
- Yahoo auto_adjust=True returns adjusted prices used even for display; actions discarded. Premium Tiingo uses raw OHLC while Polygon/Stooq have other undocumented adjustment conventions; routing silently substitutes these feeds and can ignore requested interval.
- Yahoo and Tiingo remove timezone information; cross-market date joins would leak if added naively. No available_at or information cutoff metadata. Weekday forecasts omit exchange holidays; multi-year UI selections fetch weekly/monthly bars despite day-based model horizons.
- No immutable ledger, no snapshot replay, no learned abstention/OOD/regime policy, no independent calibration set, no untouched final-test boundary.
- Free latest fundamentals, news, revised macro and factor downloads cannot reconstruct prior information without captured vintages.
- Suppressed warnings and empty failure returns can hide degraded data. Volatility intervals and synthetic test scores receive visual trust cues without coverage/calibration evidence.

## Scope guard

The audit adds no production forecast or UI changes. Phase 0 must persist reproducible baseline evidence, establish regression tests, and verify major pages before model redesign. Any unavailable real dataset/dependency is recorded as BLOCKED; synthetic tests never substitute for real OOS evaluation.
