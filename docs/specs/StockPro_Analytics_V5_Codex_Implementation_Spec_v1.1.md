# StockPro Analytics V5 — Prediction Reliability & UI/UX Redesign

Codex-ready implementation specification. Version 1.1 — 07 October 2026. Updated with Data Architecture & Source Governance.

**Core principle:** trust must be earned through auditable out-of-sample evidence, calibration, interval coverage, provenance, data freshness, and an immutable prediction history—not through visual claims.

# 1. Executive summary

StockPro Analytics already has a broad analysis surface: technical indicators, risk analytics, portfolio tools, screeners, factor analysis, news sentiment and a universal XGBoost/LightGBM checkpoint. The redesign should **preserve those useful capabilities** while replacing the current forecast mechanism and reorganizing the UI around a smaller number of high-value workflows.

- Predict **returns and return distributions**, then convert them to prices; do not train directly on raw price levels.

- Use a **three-head forecast per horizon**: direction probability, expected return, and conditional quantiles / interval.

- Keep the existing scale-free technical features, but add **market, sector, global, macro, relative-strength and event context**.

- Create **market-specific model families** (at minimum India equities and US equities). Do not treat crypto, indices and equities as one identical data-generating process merely because features are normalized.

- Make all training and evaluation **point-in-time and leakage-safe**, with purging/gaps for overlapping forward-return labels and a final untouched test period.

- Calibrate probabilities on data not used to fit the base classifier. Show reliability diagrams and Brier/log-loss metrics.

- Replace the GBM-only “confidence interval” with **model-based quantile intervals**, optionally followed by conformal calibration. Keep volatility cones only as a secondary risk reference.

- Create a **prediction ledger**: every production forecast is stamped with model version, data cutoff, features/data quality, predicted distribution and eventual realized outcome.

- Build a **confidence/abstention layer** that can downgrade or suppress forecasts when the model is out-of-distribution, models disagree, data is stale/incomplete, or historical calibration is weak.

- Redesign Streamlit around a **Forecast → Evidence → Risk → Validation** hierarchy, with progressive disclosure instead of showing 25 indicators at once.



## 1.1 Definition of success

- No V5 forecast may use future information, revised data unavailable at the timestamp, or labels whose forward windows overlap validation without being purged.

- Each supported horizon must have a documented out-of-sample record against simple baselines (zero-return/random-walk, market/sector, and current V4 logic).

- Probability statements must be empirically calibrated; “70%” must mean approximately 70% over enough comparable historical cases.

- Prediction intervals must report realized coverage as well as width. Narrow intervals are not a success if coverage is poor.

- The platform must show when the model **does not have an edge**. An abstention/no-edge state is a valid and desirable output.

- The UI may use language such as “OOS validated,” “calibrated,” or “data fresh” only when programmatically verified.



# 2. Current-state diagnosis

| Area | Current design | Priority | V5 action |

| --- | --- | --- | --- |

| Multi-day forecast | 1-day expected log return is compounded as `P0 × exp(t × r)`. | High | Replace with direct 1/5/10/20/(60)-day targets. |

| Uncertainty | GBM volatility cone is presented beside ML forecast. | High | Add quantile models; measure empirical coverage. Label GBM cone as volatility reference only. |

| Validation | Loaded checkpoint is inferred across rolling windows; unclear whether checkpoint training predates each backtest window. | Critical | Historical evaluation must train only on data available before each test period; use walk-forward/historical checkpoints. |

| Overlapping labels | 5/10/20-day forward labels share future observations. | Critical | Use purged/gapped splits so training-label windows cannot touch validation/test windows. |

| Feature scope | 56 features are primarily stock-internal technical features. | High | Add market, sector, cross-asset, macro, relative and event features with correct timestamps. |

| Data access | Core market/fundamental/news access is largely no-key/public and appears Yahoo/yfinance-centric; pages may fetch independently. | Critical | Audit every source; centralize providers, timestamps, quality, caching, provenance and explicit fallbacks. |

| Universal scope | A single pooled checkpoint is described as applicable to NSE, US and crypto. | High | Use market/asset-family checkpoints; keep scale-free features but learn different processes. |

| Confidence | Model confidence is not clearly separated from price volatility. | High | Build calibrated probability + reliability + data/OOD/ensemble confidence. |

| Sentiment | VADER + finance vocabulary is useful but shallow. | Medium | Keep as fallback; add optional local financial NLP/event extraction later. |

| UI | Broad page count and indicator density creates cognitive overload. | High | Consolidate workflows; make forecast evidence the center of the experience. |



## 2.1 What should be preserved

- Scale-free feature engineering such as `MACD / Close`, `ATR / Close`, moving-average distances and normalized volume/OBV changes.

- XGBoost + LightGBM as the initial model family; they are appropriate for heterogeneous tabular features and deploy well on Streamlit.

- Risk page concepts: VaR/CVaR, drawdown, CAPM, Monte Carlo and monthly diagnostics — but integrate risk outputs into the forecast workflow rather than isolating all of them.

- Strategy backtester and factor-analysis capabilities, subject to stronger data-snooping controls and clearer separation between strategy backtests and forecast validation.

- Rule-based assistant as a no-hallucination fallback. Later it can become a UI layer over structured model outputs rather than a source of forecasts.

- Offline-first philosophy wherever feasible, provided it does not force the product to pretend missing data is available.



# 3. Product objective and forecasting contract

```text
Forecast(ticker, as_of, horizon) -> {
  central_return, central_price, p_positive_return,
  p_outperform_market, p_outperform_sector,
  q10, q25, q50, q75, q90, intervals,
  confidence, abstain, regime, model_agreement,
  drivers, invalidation_conditions, data_quality,
  model_version, training_cutoff, feature_cutoff
}
```


## 3.1 Supported forecast horizons

| Horizon | Purpose | Recommendation |

| --- | --- | --- |

| 1 trading day | Tactical next-session direction; highest noise. | Optional/secondary in UI |

| 5 trading days | One-week short-term forecast. | Core |

| 10 trading days | Two-week swing forecast. | Core |

| 20 trading days | Approximately one trading month. | Core |

| 60 trading days | Approximately one quarter; needs more fundamental/macro context. | Phase 2 after validation |



## 3.2 Direct targets — never recursive one-day compounding

Train separate direct targets for each horizon. Do not generate 5/10/20-day forecasts by repeatedly applying the one-day output.

# 4. Forecasting architecture

```text
Point-in-time data -> feature store -> regime/OOD -> per-horizon direction + return + quantile models -> ensemble -> calibration -> confidence/abstention -> price distribution + explanations + immutable prediction ledger
```


## 4.1 Three predictive heads per horizon

| Head | Output | How to judge it |

| --- | --- | --- |

| Direction | `P(return_h > threshold)` and optionally up/neutral/down. | Calibrated probability; Brier/log loss; calibration curve. |

| Return | Expected log return for horizon `h`. | MAE/RMSE vs baselines; rank IC; sign accuracy. |

| Distribution | Conditional quantiles such as 10/25/50/75/90%. | Pinball loss; empirical interval coverage; width. |



## 4.2 Ensemble design

- Use XGBoost + LightGBM; blend with weights learned only from out-of-fold predictions.

- Persist individual outputs; use disagreement as a confidence signal.

- Use native quantile objectives where supported and enforce monotonic quantile order.



## 4.3 Market-specific model families

| Model family | Universe | Context |

| --- | --- | --- |

| India equities | NSE/BSE liquid stocks | Nifty/sector indices, India VIX, USDINR, crude, US/Asia session alignment, RBI/macro, FII/DII if available. |

| US equities | NYSE/Nasdaq liquid stocks | S&P 500/Nasdaq, VIX, US yields, DXY, sector ETFs/indices, Fed/macro. |

| Indices | Major indices | Dedicated index models or a separate task head; do not mix blindly with single-stock labels. |

| Crypto | BTC/ETH etc. | Separate 24/7 model family; different session logic and volatility process. Not part of V5 equity launch gate. |



# 5. Data and feature engineering

## 5.1 Feature families

| Family | Examples | Action |

| --- | --- | --- |

| Stock technical | Current 56 scale-free indicators; multi-horizon returns; realized volatility; gap; volume surprise; trend strength; drawdown; range compression/expansion. | Keep/extend |

| Market | Nifty 50 / S&P 500 1/5/20d returns, vol, trend, breadth, distance from moving averages, market drawdown. | Add |

| Sector | Sector return/volatility/trend; stock-minus-sector relative strength; sector-minus-market relative strength. | Add |

| Global cross-asset | Nasdaq, S&P, VIX, US 10Y changes, DXY, major Asian indices, Brent, gold; aligned by session. | Add |

| India-specific | India VIX, USDINR, Nifty Bank/IT/Energy etc.; RBI event flags; optional FII/DII flows and GIFT Nifty where legally/reliably available. | Add |

| Fundamental | Valuation ratios, growth, margins, ROE/ROCE, leverage, earnings trend; only point-in-time snapshots. | Add gradually |

| Event calendar | Earnings window, ex-dividend, corporate actions, RBI/Fed/CPI/jobs/budget/event proximity. | Add |

| News/event text | Entity relevance, finance sentiment, event type, novelty, age/decay, source quality, surprise vs expectations where data exists. | Phase 2 |

| Regime/OOD | Volatility regime, trend regime, correlation regime, feature-distance/OOD statistics, data-completeness indicators. | Add |



## 5.2 India-market session alignment

- Use timezone-aware timestamps and explicit EOD/pre-open cutoffs.

- Join exogenous features by information availability, not calendar date.

- Never forward-fill values into periods before publication.



## 5.3 Point-in-time fundamentals and corporate actions

- Use adjusted prices for labels and raw prices for display.

- Only use point-in-time fundamentals in historical tests.

- Flag corporate actions and document survivorship-bias limitations.



## 5.4 Current data-source inventory and cost posture

| Data | Current source | What it supplies | Cost / caveat | V5 treatment |

| --- | --- | --- | --- | --- |

| Prices / OHLCV / quotes | Yahoo Finance via `yfinance` | NSE (`.NS`), BSE (`.BO`), US equities, indices, FX, commodities and crypto where available. | No API key in current design; research/personal-use terms and provider reliability must be respected. | Primary historical/market feed today; add validation + provider abstraction. |

| Fundamentals / metadata | Likely Yahoo Finance via `yfinance` | P/E, EPS, beta, market cap, sector/industry and related metadata where code uses them. | No key, but field availability/schema can change; point-in-time history is not guaranteed. | Audit exact fields; never backfill present-day fundamentals into historical forecasts. |

| News | Provider must be verified in code; likely Yahoo/yfinance in the current no-key design | Headlines/article metadata used by Overview/Insights sentiment. | Do not assume source, completeness, publication timestamp or redistribution rights from README alone. | Codex must inventory exact provider and `published_at/available_at` semantics. |

| US filings | SEC EDGAR / `data.sec.gov` | 10-K, 10-Q, 8-K and XBRL/submission data used for US filing sentiment. | SEC data APIs do not require API keys; request-policy/user-agent requirements still apply. | Keep; cache responsibly and store accession/filing timestamps. |

| Fama-French factors | Kenneth French Data Library via `pandas_datareader` | Market, Size, Value, Profitability, Investment and risk-free factor data. | Public downloadable research data; version/date-cut matters for reproducibility. | Keep; persist dataset/version date in model metadata. |

| Technical / risk / signals | Local computation | RSI, MACD, ATR, VaR/CVaR, CAPM, drawdown, backtests etc. | No per-request data API cost beyond inputs. | Keep; centralize inputs so all pages use the same validated snapshot. |

| Sentiment model | VADER + local finance vocabulary | Offline news/filing polarity. | No external AI API cost. | Keep as fallback; optional local finance NLP later. |

| Forecasting | Local XGBoost + LightGBM checkpoints | Return/direction/quantile inference. | No external prediction API cost. | Keep local; version artifacts and feature schemas. |



## 5.5 Central data layer and provider abstraction

- Centralize external data behind replaceable provider adapters.

- Normalize source timestamps, availability times, symbol IDs and corporate actions before feature generation.

- Persist provider identity and snapshot/version metadata with every forecast.



## 5.6 Data quality, freshness and forecast eligibility

| Check | Requirement | Failure behavior |

| --- | --- | --- |

| Freshness | Latest required bar/quote is available for the selected forecast cutoff; report age and expected delay. | Stale critical input -> abstain or downgrade. |

| Completeness | Required OHLCV/context fields are present; missingness is measured by family. | Missing critical family -> abstain; optional family -> degrade confidence. |

| Temporal validity | `available_at <= forecast_as_of` for every feature. | Violation -> hard error in training/evaluation and forecast invalid. |

| Price sanity | Positive prices/volume semantics, no impossible OHLC ordering, suspicious jumps flagged. | Quarantine/flag until corporate action or source issue is resolved. |

| Corporate actions | Splits/dividends/bonus adjustments handled consistently between labels and display. | Unresolved action -> do not score historical error as ordinary market move. |

| Cross-source consistency | Optional secondary source or benchmark sanity checks for large discrepancies. | Flag disagreement; never auto-average unexplained discrepancies. |

| Calendar/session | Exchange calendar, timezone and session cutoff are explicit. | Do not use later US/Asian observations in an earlier India forecast. |

| Schema drift | Expected provider fields/types and feature schema hash match. | Fail closed or use an explicitly versioned compatible adapter. |



## 5.7 Forecast provenance, fallbacks and upgrade path

| Provenance field | Persist with prediction |

| --- | --- |

| Forecast identity | ticker, market, horizon, `forecast_as_of`, generated_at |

| Providers | price/market/sector/news/factor provider IDs actually used |

| Snapshot | `data_snapshot_id`, latest source timestamps, freshness/quality state |

| Model | model version, training cutoff, calibration/ensemble version |

| Schema | feature schema/version/hash and adjustment policy |

| Code | git/code version when available |



- Store provider/source and snapshot provenance with every production prediction.

- Fallback providers must be explicit and semantically compatible; stale substitution is forbidden.

- Use free/public sources for V5 validation first; premium data is a later replaceable upgrade.



## 5.8 Data-source audit required from Codex

| Inventory field | Required content |

| --- | --- |

| Call site | File/function making the network/library call. |

| Provider | Actual service/domain/library; do not infer from UI labels. |

| Instrument/field | Exact data fields consumed and their units/adjustment semantics. |

| Timestamp semantics | Observation time, publication time, timezone, `available_at`, known delay. |

| Caching | TTL/persistence, cache key and invalidation behavior. |

| Failure behavior | Retry, missing-data handling, fallback and user-visible warning. |

| Historical safety | Whether the source supports point-in-time/as-of reconstruction. |

| Usage constraints | API key/account requirement and any terms/licensing caveat known from provider docs. |

| V5 status | Keep / wrap / replace / optional / prohibited for historical backtest. |



## 5.9 News and geopolitics — later, but correctly

- Keep VADER as fallback; later add local financial NLP and event extraction.

- Prefer event type, novelty, relevance, time decay and surprise over raw sentiment.

- Model geopolitics primarily through observable cross-asset transmission channels.



# 6. Validation, leakage prevention and trust metrics

## 6.1 Split architecture

Use expanding walk-forward validation with purging/gaps for overlapping forward labels. Keep a final untouched test period.

## 6.2 Training/evaluation workflow

- Freeze a final test period; tune, blend and calibrate only on earlier walk-forward/OOS predictions; evaluate once; then train production checkpoint through the approved cutoff.



## 6.3 Baselines the model must beat

| Baseline | Definition | Why it matters |

| --- | --- | --- |

| Zero-return / random walk | Future price = current price. | Hard baseline for price MAE. |

| Historical drift | Long-run or rolling mean return. | Tests whether ML adds anything beyond drift. |

| Market/sector beta | Forecast stock from market + sector expected move. | Tests stock-specific alpha value. |

| Simple momentum/reversal | Few-feature linear/tree baseline. | Tests whether 100+ features add robust value. |

| Current V4 logic | 1-day model compounded forward. | Must be included to prove V5 improvement. |



## 6.4 Metrics by predictive head

| Area | Metrics | Interpretation |

| --- | --- | --- |

| Direction | Brier score, log loss, ROC-AUC, balanced accuracy, precision/recall by signal, calibration error/reliability curve. | Calibration and decision quality matter more than raw accuracy. |

| Return | MAE, RMSE, median AE, directional hit rate, Spearman rank IC, Pearson IC, top-vs-bottom prediction spread. | Return regression can be useful even with modest R². |

| Intervals | Pinball loss per quantile, 50%/80%/90% empirical coverage, mean interval width, coverage by regime. | An 80% interval should contain realized returns near 80% over enough cases. |

| Trading simulation | Net return after fees/slippage, Sharpe/Sortino, max drawdown, profit factor, turnover, exposure. | Secondary validation; never optimize solely for backtest P&L. |

| Stability | Metrics by year, sector, regime, cap/liquidity bucket, market, confidence decile. | Reveals where the model actually works. |



## 6.5 Probability calibration

- Calibrate direction probabilities on disjoint predictions, show reliability diagrams and sample counts, and reject calibration that degrades OOS Brier/log-loss.



## 6.6 Prediction intervals

- Use conditional quantiles as the primary interval, optionally calibrate with rolling conformal residuals, and keep GBM as a separate volatility reference.



## 6.7 Confidence and abstention

- Confidence combines calibration, ensemble agreement, regime reliability, interval width, data quality and OOD. It may abstain.



## 6.8 Prediction ledger and live audit trail

| Group | Fields |

| --- | --- |

| Identity | prediction_id, ticker, market, generated_at, as_of, horizon |

| Provenance | model_version, training_cutoff, feature_schema_version, code/git version if available |

| Inputs quality | data freshness, missing-feature count, OOD score, market regime |

| Forecast | central return/price, probabilities, quantiles, confidence, abstain flag, model disagreement |

| Explanations | top drivers and contributions; scenario/invalidation snapshot |

| Outcome | realized return/price, benchmark/sector returns, alpha, error, interval hit/miss |

| Status | pending / matured / invalidated due to bad data |



# 7. Training universe, retraining and model governance

## 7.1 Expand the pooled universe carefully

- Expand the training panel with market-specific liquid universes; evaluate weighting and recency choices out-of-sample.



## 7.2 Retraining schedule

| Cadence | Action | V5 |

| --- | --- | --- |

| Daily | Data refresh, feature computation, inference, prediction-ledger maturation. | Yes |

| Weekly | Diagnostics, drift checks, live calibration and coverage report. | Yes |

| Monthly | Candidate retrain using expanding history; compare challenger vs champion on fixed validation protocol. | Recommended |

| Event-driven | Retrain only if schema/data/provider changes or drift thresholds are breached. | Optional |

| On every page load | Training. | No — inference only |



## 7.3 Champion/challenger model governance

- Version every model; use champion/challenger promotion based on the frozen OOS suite; keep rollback artifacts.



## 7.4 Drift and out-of-distribution monitoring

- Monitor feature/prediction drift, missingness, calibration and interval coverage; severe OOD should force low confidence or abstention.



# 8. Full UI/UX redesign

## 8.1 Information architecture

| Top-level area | Contains | Role |

| --- | --- | --- |

| Dashboard | Market snapshot, watchlist, active forecasts, model/data health, recent matured predictions. | Default landing page. |

| Research | Overview + Technical + Fundamentals + Factors + News + Compare + Screener as tabs/subpages. | Consolidates many current pages. |

| Forecast | Core price-distribution forecast, drivers, scenarios, invalidation, historical analogues, model disagreement. | Hero product. |

| Validate | Model card, OOS performance, calibration, interval coverage, prediction ledger, strategy backtests. | Trust center. |

| Risk & Portfolio | Portfolio, VaR/CVaR, drawdown, correlations, exposure, scenario impact. | Action/risk layer. |

| Markets | Global indices, sectors, breadth, VIX, macro/cross-asset dashboard. | Context layer. |



## 8.2 Visual design system

| Token | Recommendation | Reason |

| --- | --- | --- |

| Base | Deep navy `#0B1220` and slate surfaces; optional light theme. | Institutional, neutral. |

| Text | Off-white/light slate in dark mode; near-black in light mode. | High readability. |

| Accent | Blue for navigation/selection; green/red only for actual positive/negative financial states. | Avoid “casino” feel. |

| Warning | Amber for stale data, low confidence, model-health issues. | Operational risk cue. |

| Typography | Aptos/Inter/system sans; tabular numerals for prices/metrics; 3–4 clear size levels. | Fast scanning. |

| Cards | Thin borders, subtle surface contrast, almost no shadows. | Serious and clean. |

| Charts | Neutral price line/candles; uncertainty bands more visually dominant than arbitrary indicator colors. | Forecast-first. |

| Accessibility | Never encode up/down only by red/green; include arrows, labels and signs. | Color-blind safe. |



## 8.3 Trust cues that are allowed

- Expose timestamps, model version/training cutoff, OOS/calibration status, interval coverage, sample sizes, data completeness and full prediction history.



### 8.3.1 Trust cues that are forbidden

- Forbid unsupported accuracy claims, opaque AI-buy badges, fake confidence and selective presentation of wins.



## 8.4 Forecast page wireframe

Hero: ticker + freshness + model version; central price/return + calibrated probability + reliability; horizon tabs; forecast distribution chart; drivers; trust metrics; then Scenario Lab, invalidation, analogues and ledger.

## 8.5 Forecast page components

| Component | Content | Purpose |

| --- | --- | --- |

| A. Header | Ticker, market, live/current price, session state, data cutoff, refresh, model version. | Always visible. |

| B. Forecast hero | Central future price, expected return, P(up), P(outperform), 50/80% intervals, confidence grade. | Primary decision surface. |

| C. Horizon selector | 1D / 5D / 10D / 20D / 60D. | Switches all downstream evidence consistently. |

| D. Distribution chart | Historical price + central path/endpoint + shaded quantile bands + realized outcomes for matured forecasts. | Visual uncertainty. |

| E. Drivers | SHAP/feature-attribution groups with contribution direction; distinguish model association from causal claim. | Why. |

| F. Market decomposition | Expected market + sector + stock-specific alpha/event components. | Context. |

| G. Model agreement | XGB vs LGBM vs heads; highlight disagreement. | Confidence evidence. |

| H. Invalidation | Conditions/scenario changes that materially reduce forecast probability. | When not to trust it. |

| I. Scenario Lab | User changes global/market assumptions; recompute using scenario feature overrides where statistically valid. | What-if analysis. |

| J. Historical analogues | Nearest historical feature states; realized outcome distribution; clearly label small samples. | Concrete evidence. |

| K. Trust panel | OOS metrics, calibration bin, interval coverage, sample counts, drift/OOD, data quality. | Auditability. |

| L. Prediction ledger | Past forecasts and realized errors; filter by ticker/horizon/model. | Accountability. |



## 8.6 Progressive disclosure rules

- First viewport answers forecast center, uncertainty, reliability and timestamp; critical warnings are never hidden; indicators move to Research unless they actually drive the forecast.



## 8.7 Streamlit implementation guidance

- Use st.navigation, config-based theming, cache_data/cache_resource, fragments for independently updating panels and reusable UI components.



# 9. Proposed codebase structure

```text
See DOCX for proposed file tree; key modules separate data/alignment, feature families, prediction targets/splits/heads, calibration/intervals/confidence, validation, explanations and ledger.
```


## 9.1 Core schemas

## 9.2 Model metadata contract

# 10. Codex implementation plan

| Phase | Scope | Exit criterion |

| --- | --- | --- |

| Phase 0 — Baseline & guardrails | Freeze V4 behavior, add tests, capture current metrics/screens, create model/data metadata schemas. | No user-visible forecast changes. |

| Phase 1 — Leakage-safe data pipeline | Point-in-time joins, session alignment, target builder, purged/expanding splits, final-test boundary. | Must pass no-future-data tests. |

| Phase 2 — Direct multi-horizon core | 1/5/10/20-day direction + return models, pooled market-specific panel, V4 comparison. | Remove multi-day one-day compounding from production path. |

| Phase 3 — Conditional distributions | Quantile models, quantile-order correction, interval coverage reporting, optional conformal calibration. | Replace GBM as primary forecast interval. |

| Phase 4 — Context features | Market/sector/global/macro/relative features; regime and OOD. | Run ablation tests to prove each feature family adds OOS value. |

| Phase 5 — Calibration & confidence | Probability calibration, model agreement, confidence components, abstention/no-edge logic. | No “high confidence” without evidence. |

| Phase 6 — Prediction ledger & trust center | Immutable forecasts, maturation jobs, Model Card/Validate page, calibration/coverage charts. | Trust evidence visible. |

| Phase 7 — UI redesign | New IA, Forecast page, Dashboard, consolidated Research, design tokens, reusable components. | Remove clutter; preserve full tools under progressive disclosure. |

| Phase 8 — News/event intelligence | Structured event pipeline, local finance NLP option, novelty/decay/surprise, geopolitics via transmission channels. | Only after core forecast beats baselines. |

| Phase 9 — 60-day/fundamental expansion | Quarterly horizon, stronger fundamental/earnings features, additional model family experiments. | Separate validation gate. |



## 10.1 Phase 0 — exact tasks

- Phase 0: freeze V4, add schemas/timestamps/as-of reconstruction, and benchmark it against simple baselines.



## 10.2 Phase 1–2 — minimum viable V5 model

- MVP V5: direct multi-horizon XGB/LGB direction + return heads on purged walk-forward India data; V4 compounding remains only as baseline.



## 10.3 Phase 3–5 — make the forecast trustworthy

- Trust phases add quantiles, calibration, OOD/confidence/abstention and feature-family ablations.



## 10.4 Phase 6–7 — trust center and UI makeover

- Build Validate trust center and redesigned Forecast page; consolidate navigation and reusable UI.



# 11. Acceptance criteria and release gates

## 11.1 Data integrity gates

- Automated test proves every feature row has `available_at <= forecast_as_of`.

- Automated test proves no training label interval overlaps a validation/test interval for each horizon.

- All corporate-action-adjusted labels are reproducible.

- Historical inference can be run with `--as-of` without silently fetching current fundamentals/news into the feature row.

- Provider failures/stale values produce an explicit quality flag rather than forward-filled fake freshness.

- All external provider calls are represented in `docs/DATA_SOURCES.md` / provider registry; UI/model modules do not bypass the provider layer.

- Every persisted production forecast records provider IDs, snapshot/data cutoff, freshness/quality state and model/feature/code versions.

- Any provider fallback is visible in provenance and is rejected when adjustment/timestamp semantics are incompatible.



## 11.2 Model performance gates

- Compare V5 to V4 and naive baselines on frozen final-test data; require balanced improvements across calibration, errors/rank and interval coverage, with regime/sector breakdowns.



## 11.3 Product/UI gates

- UI must expose timestamp/model provenance, forecast distribution and reliability first; critical warnings are always visible; trust labels link to real evidence.



## 11.4 Suggested promotion decision

# 12. Required automated tests

| Test | Requirement |

| --- | --- |

| `test_no_future_leakage.py` | Construct synthetic timestamps where a feature arrives after as-of; assert exclusion/error. |

| `test_label_purging.py` | For every fold/horizon, assert train `[t,t+h]` intervals do not intersect validation intervals. |

| `test_session_alignment.py` | Verify US/India/Asia samples map to correct information cutoffs. |

| `test_feature_point_in_time.py` | Historical build uses only source rows available at that time. |

| `test_quantile_order.py` | Ensure returned quantiles are monotonic after post-processing. |

| `test_calibration_disjoint.py` | Assert calibrator inputs do not come from rows used to fit the corresponding base prediction. |

| `test_prediction_schema.py` | All forecast outputs complete, bounded probabilities, valid horizons/prices. |

| `test_model_metadata.py` | Artifacts cannot load if schema hash/version mismatch. |

| `test_abstention.py` | Stale/OOD/missing critical inputs force downgrade or abstain. |

| `test_v4_not_used_for_v5_multiday.py` | 5/10/20-day production inference never calls one-day compounding path. |

| `test_provider_contract.py` | Each provider returns normalized source/source_timestamp/available_at/fetched_at/quality metadata and canonical fields. |

| `test_provider_fallback.py` | Fallback is explicit; stale/incompatible provider data cannot be silently substituted. |

| `test_data_snapshot_reproducibility.py` | A persisted prediction can reload the same snapshot/schema/provider metadata required to reproduce its features. |

| `test_ui_smoke.py` | All top-level pages load with mock data and missing optional providers gracefully. |



# 13. Explicit “do not do this” list for Codex

- Do not replace XGBoost/LightGBM with LSTM/Transformer before fixing targets, context and validation.

- Do not use `train_test_split(shuffle=True)` anywhere in forecast model validation.

- Do not train one full-history checkpoint and then claim historical rolling inference from that checkpoint is out-of-sample.

- Do not tune hyperparameters, confidence thresholds, feature sets or ensemble weights on the final test period.

- Do not use present-day fundamentals/news in historical feature rows.

- Do not call GBM cone width “model confidence.”

- Do not treat SHAP/feature attribution as causal proof. Label it “model contribution/association.”

- Do not add hundreds of indicators without ablation evidence. More correlated features can increase overfitting.

- Do not mix crypto, indices, India equities and US equities into one “universal” model simply because technical features are scale-free.

- Do not silently fall back to the small single-ticker model and show the same visual trust treatment as the validated production checkpoint.

- Do not optimize the strategy backtester until the forecast itself is frozen; repeated strategy search can overfit the backtest.

- Do not make marketing claims such as “best prediction,” “institutional accuracy,” or “most trustworthy” from aesthetics. Let the live ledger and OOS metrics make the case.

- Do not call Yahoo/yfinance data “real-time,” “exchange-grade,” or “production-grade” unless the actual source/terms/timestamps prove that claim.

- Do not fetch the same external data independently from multiple pages. Use the provider/snapshot layer so analysis and prediction share one validated view.

- Do not silently merge or fallback across providers with different adjusted-price, timestamp, interval or corporate-action semantics.



# 14. Optional accuracy improvements after the core is proven

| Idea | What it adds | When |

| --- | --- | --- |

| Cross-sectional ranking | Predict rank/alpha among stocks each day in addition to absolute return. | Often easier to evaluate and useful for screeners/portfolio construction. |

| Meta-labeling | Primary model proposes direction; secondary model estimates whether the signal is worth acting on. | Good fit for abstention/no-trade layer. |

| Mixture of experts | Different submodels for trend/high-vol, mean-reverting/low-vol, event regimes. | Only if regime-specific OOS evidence supports it. |

| Analyst estimate surprise | Earnings/revenue actual-vs-consensus and revision momentum. | Potentially high value, but needs a reliable point-in-time data source. |

| Derivatives features | Implied volatility, skew, open interest, put/call, futures basis. | Powerful for liquid names; data availability/licensing may be limiting. |

| FII/DII and flows | Institutional flow and market breadth/participation. | India-specific contextual signal. |

| Adaptive conformal intervals | Rolling calibration under distribution shift. | Improves uncertainty honesty if monitored correctly. |

| CPCV / multiple backtest paths | Distribution of backtest paths rather than one historical path. | Useful for robustness after the simpler expanding purged workflow works. |

| Local finance NLP | FinBERT-class sentiment and event classifier. | Add only when news timestamps/source quality are reliable. |

| Dedicated feature store | Parquet/DuckDB + schema/versioned snapshots. | Improves reproducibility and training speed as dataset grows. |



# 15. Codex master instruction

```text
You are upgrading StockPro Analytics to V5. Treat this specification as the source of truth.

PRIMARY GOAL
Build the most defensible and auditable forecast system possible within the existing Python/Streamlit/XGBoost/LightGBM architecture. Do not optimize for impressive-looking predictions. Optimize for point-in-time correctness, leakage-free validation, calibrated uncertainty, reproducibility and transparent UI evidence.

IMPLEMENTATION RULES
1. Work in phases. Do not rewrite the whole app at once.
2. Preserve current analytics pages until their replacements pass smoke tests.
3. First freeze V4 behavior and add regression tests.
3A. Before model changes, audit every external data call and create `docs/DATA_SOURCES.md`; then centralize access behind provider adapters that emit source/timestamp/freshness/quality/provenance metadata. Wrap the current Yahoo/yfinance path first and keep free/public sources as the V5 default.
4. Replace multi-day `exp(t * r_1d)` production forecasting with direct 1/5/10/20-day targets.
5. Build direction, return and quantile heads for each horizon using XGBoost + LightGBM.
6. Use market-specific model families. Start with India equities; keep US separate.
7. Every feature must be point-in-time. Track source timestamp and available_at.
8. Implement expanding walk-forward validation with label purging/gaps for overlapping forward-return windows, plus a frozen final test period.
9. Fit preprocessing, feature selection, model tuning, ensemble weights and calibration only on training/OOF data. Never use final-test data for decisions.
10. Add market, sector, global/macro and relative-strength features in controlled groups. Run ablation tests; keep only groups with stable OOS benefit.
11. Calibrate direction probabilities on disjoint predictions. Build conditional quantile intervals and report empirical coverage.
12. Confidence must combine calibrated probability strength, ensemble agreement, regime reliability, interval width, data quality, OOD and sample support. It must be able to abstain.
13. Persist model metadata and every production prediction. Mature predictions later with realized outcomes.
14. Redesign the app around Dashboard / Research / Forecast / Validate / Risk & Portfolio / Markets. The Forecast first viewport must show price distribution, probability, reliability, timestamp and model provenance—not an indicator wall.
15. Never hide stale-data, fallback, OOD, low-confidence or failed-validation warnings.
16. Do not add LSTM/Transformer until the V5 tree-based baseline passes the frozen validation suite.
17. Do not claim accuracy/trust that is not evidenced by stored OOS metrics.

DELIVERY FORMAT FOR EACH PHASE
- Summarize files changed.
- Explain data/model assumptions.
- Show tests added and test output.
- Show before/after OOS metrics where the phase changes forecasting.
- List known limitations and any data source that is not truly point-in-time.
- Do not silently weaken validation just to improve metrics.

```


# 16. Research / implementation references

| Reference | Use in V5 | Link |

| --- | --- | --- |

| Scikit-learn — TimeSeriesSplit | Time-ordered cross-validation and `gap` support. | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html |

| Scikit-learn — Probability calibration | Why calibrators should use data independent of base-model fitting; calibration methods and reliability concepts. | https://scikit-learn.org/stable/modules/calibration.html |

| XGBoost — Quantile regression | Native `reg:quantileerror` and quantile-alpha support in current XGBoost releases. | https://xgboost.readthedocs.io/en/latest/parameter.html |

| XGBoost — Prediction intervals example | Worked quantile/expectile interval example and quantile-crossing caveat. | https://xgboost.readthedocs.io/en/latest/python/examples/prediction_intervals.html |

| Adaptive Conformal Predictions for Time Series (PMLR, 2022) | Research on adaptive conformal inference under time-series dependence/distribution shift. | https://proceedings.mlr.press/v162/zaffran22a.html |

| Streamlit — st.Page / st.navigation | Preferred current mechanism for flexible multipage navigation. | https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation |

| Streamlit — theming | Theme configuration and light/dark theme support. | https://docs.streamlit.io/develop/concepts/configuration/theming |

| Streamlit — caching | Use `st.cache_data` for data/computation and `st.cache_resource` for model/resource objects. | https://docs.streamlit.io/develop/concepts/architecture/caching |

| Streamlit — fragments | Independent reruns for selected UI regions. | https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment |

| yfinance documentation / disclaimer | Current project market-data wrapper; documents use of Yahoo public APIs and research/personal-use caveat. | https://github.com/ranaroussi/yfinance/blob/main/doc/source/index.rst |

| SEC — EDGAR APIs | US submissions/XBRL APIs; SEC states `data.sec.gov` requires no authentication/API key. | https://www.sec.gov/search-filings/edgar-application-programming-interfaces |

| Kenneth French Data Library | Public downloadable Fama-French factor datasets and historical archives. | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |

| Purging/embargo concept | Forward-return labels overlap; purging removes training labels whose information intervals overlap test intervals. This is a finance-specific extension beyond ordinary chronological splitting. | https://pmc.ncbi.nlm.nih.gov/articles/PMC9521884/ |



# 17. Final product principle

The headline remains a future price. The moat is the evidence: point-in-time provenance, OOS validation, calibration, interval coverage, regime/sample context, failure conditions and an immutable live record.
