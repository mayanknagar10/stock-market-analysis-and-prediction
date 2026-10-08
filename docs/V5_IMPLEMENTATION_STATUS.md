# StockPro V5 implementation status

Updated: 08 October 2026. Branch: v5-development. Governing sources: the user's phased request and supplied v1.1 spec. [Master task board](V5_TASK_BOARD.md) is the current progress checklist; notes below preserve the historical phase record.

**Final state: research delivery complete; numerical gate FAILED; production definition of success unmet.** No implementation task is currently in progress. News/NLP expansion is performance gated, browser visual/accessibility QA is not done, PIT certification and production promotion remain blocked. The final period was consumed once; do not tune or rerun it.

## Architecture and phase exits

Additive core/data, core/validation and core/forecasting modules preserve original V4 source/artifacts and existing analytics. Source adapters -> immutable snapshots/quality -> causal technical/context features -> separate India/US direct heads -> disjoint calibration/OOD/abstention -> immutable ledger -> Forecast/Validate. Explicit capture/training/resolution jobs stay outside ordinary page loads.

| Phase | Final state | Main files | Evidence / limits |
|---|---|---|---|
| 0 Audit and V4 freeze/baseline | DONE | docs/REPOSITORY_AUDIT.md; DATA_SOURCES.md; scripts/evaluate_v4.py; reports/baseline | Original V4 bytes/source hashes retained; synthetic checkpoint separated from real-market diagnostics |
| 1 Temporal validation | DONE — RESEARCH | core/validation/*; core/data/contracts.py, snapshots.py, quality.py | Availability/revision, timezone precision, purging, scaler and snapshot tests; real historical certification blocked |
| 2 Direct heads | DONE — RESEARCH | core/forecasting/heads.py, panel.py, training.py, inference.py; scripts/train_v5.py | Direct four-horizon India/US models; development evidence persisted; no final advantage claim |
| 3 Context/data | DONE — RESEARCH | core/data/providers.py, runtime.py, cleaning.py, legacy_*; core/forecasting/context.py; config/context_sources.json | Central provider operations; dated snapshots; placeholder cleanup; development ablations; no fake reference filling |
| 4 Distributions | DONE — RESEARCH | core/forecasting/distribution.py | Independent direction and native q10/q25/q50/q75/q90 heads; pinball/coverage/width metrics |
| 5 Calibration/trust | DONE — RESEARCH | core/forecasting/trust.py; training.py | Four separately purged blocks; harmful calibrations rejected; all research LOW/abstained |
| 6 Relative/regime | DONE — RESEARCH | core/forecasting/relative.py; context.py; final metrics | Session-aligned auxiliary heads and regime breakdowns where available; limited dated membership/history |
| 7 Ledger | DONE — CORE | core/forecasting/ledger.py; scripts/resolve_predictions.py | Atomic append-only forecasts/outcomes, integrity, known-origin maturity; optional alpha references unavailable in ordinary resolver |
| 8 Scenarios/explanation | DONE — CORE | core/forecasting/scenarios.py; inference.py | Rebuild source features; native SHAP associations; past-matured analogues; abstaining forecasts have no actionable edge |
| 9 News/events | DEFERRED — PERFORMANCE GATED | Existing core/sentiment.py preserved; core/data/events.py interface | Expansion requires a proven core; no FinBERT, unreliable scraping or claimed event coverage |
| 10 Registry/hardening | DONE — RESEARCH | core/forecasting/registry.py, service.py; models/v5 | Complete native checksums, all cutoffs, quarantine; production rejects; durability/auth migrations outstanding |
| 11 UI | DONE — FUNCTIONAL | app.py; pages/dashboard.py, forecast.py, validate.py; utils/ui_v5.py, charts.py, helpers.py | Existing/new page tests, source/semantic review; actual browser visual/a11y checks unavailable |
| 12 Final/docs | DONE | scripts/evaluate_final.py, render_final_report.py; reports/V4_VS_V5.md; docs/V5_COMPLETION_REPORT.md, V5_ARCHITECTURE.md, V5_DEPLOYMENT.md; README.md | Final lock/one-shot result; 122 tests passed; release remains research-only |

Phase commits: c9d1255 (audit/baseline), 53b4e90 (ignore restoration), b7cc29d (temporal infrastructure), 197df9d (direct heads), e535ecd (providers/context), 71377b2 (distributions/calibration/relative), cd5b0d3 (ledger/scenarios/hardening), 22c3f6f (UI), 21efae2 (final protocol), plus the protocol lock and final closeout commits. Git history records exact file lists; no force push, merge or deployment was performed.

## Final evidence and deviations

The active candidate is research-20261008T190859; all eight artifacts share one frozen training source hash. research-20261007T170953 was quarantined for blending/calibration row reuse; research-20261007T173345 for source placeholder/session quality. Replacement fits and reports remain stored; no failure history was deleted.

Core functionality and safety contracts were validated before the trust UI switch. Production performance validation has failed, so UI displays experimental scores, source uncertainty and abstention. Phase 9 enhancement was deliberately deferred under spec §10's requirement that the core first beat baselines. This deviation was documented; it was not presented as finished NLP work.

Final test begins 2025-10-01 UTC; labels mature before the 2026-10-07 original capture cutoff. Gates: MAE 1/8 vs required 6, RMSE 0/8 vs 6, Brier 2/8 vs 6, coverage 7/8 vs 6, accepted calibration 3/8 vs 8. Gate FAILED. [Complete metrics](../reports/V4_VS_V5.md) include sector/regime/year/stock breakdowns and matched V4 recursion. Research abstention 100%; production remains disabled.

Final regression: 122 passed, eight preserved UTC display deprecation warnings, 33.54 seconds. Temporal/snapshot/provider/target/split/scaler/head/distribution/calibration/registry/ledger/scenario/analogue tests and all 13 existing page initializations passed. Additional Forecast route and semantic fixtures passed. Final compile/startup/result-integrity evidence: reports/validation/V5_VERIFICATION.json. No configured lint/type pipeline exists. Browser runtime had no available browser, so responsive/pixel/accessibility QA is NOT claimed.

Performance: ordinary inference remains checkpoint-only, with versioned model caches and 300-second dated data cache. All-horizon writes are atomic. No measured latency improvement claim over V4; native artifact history is approximately 136 MB. Original indicators/models and artifact bytes remain protected by the V4 freeze. Git text normalization is disabled for frozen code/artifacts/config/results; all 252 staged Git blobs match their exact frozen hashes, with no functional core/model changes.

## Historical notes (superseded by the final table)

## Migration risks

Preserve models/universal_* files, legacy analytics and local user data. JSON auth writes are concurrency-prone; any storage migration needs backup and rollback. Ephemeral Streamlit hosting loses users/snapshots/ledger unless a durable volume is supplied. Native boosters avoid loading arbitrary pickle artifacts; validate schemas and file integrity before inference. New exchange-session policies may differ from legacy weekday dates; surface the change. Present-day macro/news/fundamentals must remain research-only without true historical vintages. Current-universe survivorship bias must stay in model cards.

## Phase record

Phase 0 audit inventory and frozen-checkpoint diagnostic COMPLETE (20 source/call families; 72 real-equity diagnostic observations per horizon). 14 metric/preservation/loader tests and 13 page smoke checks pass. Market/sector baseline extension and true historical checkpoint lineage remain unresolved; no claim of OOS V4 accuracy. Initially missing runtime dependencies installed into ignored .deps, leaving global Python unchanged. Standard exec helper fails setup; approved execution works. Yahoo direct HTTP endpoint returned 429, while yfinance history route returned 1241 RELIANCE.NS rows. Source failure must not be mistaken for missing market data. No production model/UI architecture changes yet. The Windows LightGBM text file is CRLF but declares LF byte offsets; diagnostic uses an explicitly normalized in-memory copy with integrity checks. Original files remain unchanged. No OOS superiority claim.

## Historical phase-0 exit note

Phase 0 exit: 27 automated checks passed, Streamlit health endpoint returned ok with --global.developmentMode=false. No original forecast/UI files changed; all checkpoint bytes unchanged. Per-head unavailable V4 statistics are null. Diagnostic evidence is not historical OOS evidence. Phase 1 can add independent safety infrastructure; promotion still requires reconstructible as-of data and legitimate historical fits.

User follow-up: no historical archives/checkpoints exist; user instructed “do whatever is best” after the research-only continuation proposal. Continue remaining implementation as labeled experimental research using frozen retrospective snapshots. Do not set point_in_time_verified=true or promote production. Strict point-in-time certification remains BLOCKED. This is a documented scope adaptation, not a weakened production acceptance gate.

Phase 1 implementation: core/validation targets/splits/alignment/preprocessing/model contracts; core/data provider/snapshot/health contracts. Fail-closed verification flags, defensive snapshot copies, named timezone/precision replay, late-revision alignment and pooled label interval purging tested. Category snapshots explicitly rejected until category schema exists. Current retrospective Yahoo snapshots remain ineligible for point-in-time certification. Research-only modeling authorized by latest user steering; all production gates remain enforced.

Master progress board requested by user: docs/V5_TASK_BOARD.md. Current task is integrated research training/evidence persistence. Phase 2 purged three-fold regressors completed for India/US 1/5/10/20 sessions; persisted V5_DIRECT_REGRESSION.json/CSV. Improvement over simple baselines is inconsistent and is not claimed.
