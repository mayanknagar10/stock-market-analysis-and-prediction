# StockPro V5 — master task board

Updated: 08 October 2026. This is the single progress checklist for the upgrade, also displayed on Dashboard.

**Current task: research closeout complete. Next dependent task: obtain verified historical data and define a new untouched holdout before further model development.**

**Release state: RESEARCH ONLY — final performance gate FAILED; production promotion BLOCKED.** The final period has been consumed once. It is closed to further model tuning.

| Task | Status | Evidence / remaining action |
|---|---|---|
| 0. Repository audit and source inventory | DONE | Audit; 20 source/call families; frozen V4 hashes |
| 0. Repeatable V4 baseline | DONE | Six equities; four horizons; separate synthetic-checkpoint diagnostics and cost-adjusted strategy reports |
| 1. Temporal validation infrastructure | DONE — RESEARCH | Availability/vintage joins, session/timezone guards, purged splits, snapshots and schema tests |
| 2. Direct market-specific regressors | DONE — RESEARCH | Independent XGB/LGB 1/5/10/20-session models; no final superiority demonstrated |
| 3. Central data layer and context | DONE — RESEARCH | Source governance, reference snapshots, development-only ablations; historical availability unverified |
| 4. Direction and quantile heads | DONE — RESEARCH | Eight family/horizon pairs; native five-quantile heads; 50%/80% bands |
| 5. Calibration, OOD, confidence, abstention | DONE — RESEARCH | Four disjoint purged blocks; 3/8 calibrations accepted on development; all forecasts LOW/abstained |
| 6. Regimes and relative forecasts | DONE — RESEARCH | Descriptive regimes, benchmark/sector heads where references support them; final breakdowns persisted |
| 7. Immutable ledger and outcomes | DONE — CORE | Atomic append-only batches; hashes; known origins and independent maturity calendars; optional reference alpha |
| 8. Scenarios, invalidation, analogues | DONE — CORE | Source perturbations rebuild features; native SHAP; only matured analogue outcomes |
| 9. News/events/NLP enhancement | NOT DONE — PERFORMANCE GATED | Existing VADER preserved; event interface implemented; expansion deferred because core failed its gate |
| 10. Registry and research hardening | DONE — RESEARCH | Native artifacts, complete checksums, source identity, cutoff enforcement and quarantine history; production disabled |
| 11. Navigation and Forecast/Validate UI | DONE — FUNCTIONAL | Source/semantic and AppTest checks; desktop/mobile/accessibility browser QA remains NOT DONE |
| 12. Final regression, comparison and docs | DONE | 122 tests passed; locked one-shot comparison; README, architecture, deployment and completion reports |
| Verified historical availability/revisions | BLOCKED | User confirmed no archives; structural data health does not establish PIT eligibility |
| Verified exchange calendar | BLOCKED FOR CERTIFICATION | Independent observed calendars are research proxies |
| Production promotion | BLOCKED | Failed numerical gate plus unverified historical lineage; no production inference path |
| Wider universe / 60D / crypto/index/FX | NOT DONE — SEPARATE GATES | Six-stock research universe; equity families only; dated memberships needed |

No implementation task is currently in progress. The remaining work above is explicitly deferred, unsupported or blocked; it has not been marked complete.

Final gate: MAE improvements 1/8 (required 6), RMSE 0/8 (6), Brier below 0.25 2/8 (6), 80% coverage tolerance 7/8 (6), accepted calibration 3/8 (8). Actionable forecast coverage 0%; research abstention 100%. These results do not support real-world decision reliance.

Candidate research-20261008T190859 is retained as the final research artifact. Earlier candidates remain visible, including quarantined research-20261007T170953 (blend/calibration row reuse) and research-20261007T173345 (placeholder/session-source quality). Raw snapshots and failed reports were preserved.

[Comparison](../reports/V4_VS_V5.md) · [Completion report](V5_COMPLETION_REPORT.md) · [Detailed phase record](V5_IMPLEMENTATION_STATUS.md) · [Verification](../reports/validation/V5_VERIFICATION.json).
