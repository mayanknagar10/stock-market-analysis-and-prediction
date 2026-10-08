# StockPro V5 — master task board

Updated: 08 October 2026. This is the single progress checklist for the V5 upgrade.

**Current task: final regression, the one-time frozen comparison, and completion documentation.**

**Release state: RESEARCH ONLY.** Historical point-in-time certification and production promotion remain blocked because availability/revision archives are absent. The final test period remains sealed.

| Task | Status | Evidence / next action |
|---|---|---|
| 0. Repository audit and source inventory | COMPLETE | Repository audit, 20 source families, frozen V4 hashes |
| 0. Repeatable V4 baseline | COMPLETE | Six equities; 1/5/10/20D diagnostics and cost-adjusted strategy reports; synthetic checkpoint limitations disclosed |
| 1. Leakage-safe validation infrastructure | COMPLETE | Purging, horizons, availability/vintage joins, timezone/precision, snapshot and schema guards; 78 tests passed at phase exit |
| 2. Direct market-specific return regressors | COMPLETE — RESEARCH CORE | XGB + LGB per horizon; three purged development folds; results persisted; no consistent baseline advantage |
| 3. Central data layer and context | COMPLETE — RESEARCH | Provider calls centralized; contextual snapshots and pre-calibration ablation evidence persisted |
| 4. Direction and quantile forecasts | COMPLETE — RESEARCH | Independent direction and conditional quantile heads trained for all eight family/horizon pairs |
| 5. Calibration, confidence, OOD, abstention | COMPLETE — RESEARCH | Four disjoint purged blocks; unsupported calibrations rejected; all forecasts remain LOW/abstained |
| 6. Regimes and relative/alpha forecasts | COMPLETE — RESEARCH | Market/volatility regime breakdowns and market/sector excess-return heads persisted where available |
| 7. Immutable prediction ledger | COMPLETE — CORE | Append-only forecasts/outcomes, source hashes, known origins and independent maturity calendars tested |
| 8. Scenarios, invalidation, analogues | COMPLETE — CORE | Source perturbations rebuild features; native SHAP and matured historical analogues tested |
| 9. News and corporate events | NOT STARTED / PERFORMANCE GATED | Existing VADER preserved; enhancements follow validated forecasting evidence |
| 10. Model registry and hardening | COMPLETE — RESEARCH | Native registry, complete checksums and quarantine implemented; end-to-end hardening continues |
| 11. Forecast, Validate, navigation/UI | COMPLETE — FUNCTIONAL | Forecast/Validate/Dashboard connected; source/semantic review passed; browser visual QA unavailable |
| 12. Final regression, comparison and docs | IN PROGRESS | Regression checks and protocol preflight underway; execute the final period once after locking |
| Point-in-time certification | BLOCKED | No historical availability/revision archives; user confirmed absence |
| Production promotion | BLOCKED | Requires verified data plus successful untouched-test evidence |

Task order follows the supplied specification. Native registry and evidence persistence are implemented; final verification does not promote a checkpoint or weaken the production gate.

Detailed phase records: [V5_IMPLEMENTATION_STATUS.md](V5_IMPLEMENTATION_STATUS.md).

Review finding retained for audit: research-20261007T170953 was quarantined for blending/calibration row reuse. Replacement research-20261007T173345 uses separate purged blocks; original artifacts/reports remain visible and no final-test rows were evaluated.

Current data-quality finding: Yahoo equity histories include flat zero-volume placeholder bars on dates absent from independent positive-volume session evidence. Those rows are explicitly quarantined while raw snapshots remain unchanged; unresolved real-session rows block affected input windows/labels. Candidate research-20261007T173345 is retained and quarantined; replacement research-20261008T190859 is active. All eight artifacts share one frozen training-source hash.

Resumed verification: 115 tests passed; Forecast route initialized without errors. Source-level UI review required explicit 50%/80% bands and calibration/source/OOD status before values; those corrections are implemented. Browser-based visual/accessibility checks remain unavailable and are not claimed as passed.

Final source/semantic UI review: both interval bands, calibration state and source/OOD warnings now precede outputs; accepted/rejected fixtures render without errors. Browser visual/accessibility QA remains unavailable. Candidate research-20261008T190859 is the final research candidate; protocol locking/evaluation is the current task.
