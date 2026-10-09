# Initial post-V5 research delivery

Completed: 09 October 2026. Branch: research/prospective-20261008. Study: prospective-india-20261008-v1. Research only; no production promotion or successor final fit.

## Delivered and preserved

Original V5 is retained at 26c09e0 / research-20261008T190859. All 1,052 protected original files, including V5 reports/models/locks/markers/snapshots/ledger/quarantines and frozen core sources, match their sealed hashes. The consumed final period was not opened, reevaluated or used to choose new features, models, targets, calibration, thresholds or architecture. Original final failure remains visible.

The eight initial deliverables are available:

| Deliverable | Evidence |
|---|---|
| Registered protocol | [NEXT_RESEARCH_PROTOCOL.md](NEXT_RESEARCH_PROTOCOL.md); config/research/prospective-india-20261008-v1/PROTOCOL.json |
| Fixed prospective universe | config/research/prospective-india-20261008-v1/UNIVERSE.json and exact official Nifty100 CSV:100 securities,17 industries, ISIN/source/membership/acquisition identity; no performance/provider-success filtering |
| Prospective archive/ledger verification | Immutable raw inputs/run envelopes/forecast records, separate append-only shadow ledger and explicit resolver; public capture/shadow/verification summaries under reports/post_v5 |
| V5 failure analysis | [Corrected analysis](../reports/post_v5/development-20261009T083921-00e775cf/V5_FAILURE_ANALYSIS.md) plus [exact native train replay](../reports/post_v5/V5_NATIVE_TRAIN_ADDENDUM.md) and [calibration diagnostic](../reports/post_v5/V5_CALIBRATION_ADDENDUM.md) |
| Strong simple baselines | [Corrected comparison](../reports/post_v5/development-20261009T083921-00e775cf/BASELINE_COMPARISON.md) |
| Targets/decomposition/cross-section | [Target results](../reports/post_v5/development-20261009T083921-00e775cf/TARGET_FORMULATIONS.md), paired cohorts and sector/interactions/held-out-industry experiments |
| Feature groups/stability | [Ablations](../reports/post_v5/development-20261009T083921-00e775cf/FEATURE_ABLATIONS.md), train-only pruning, source-mask and numerical IC audits |
| Proposed architecture | [Conditional proposal](../reports/post_v5/development-20261009T083921-00e775cf/NEXT_CANDIDATE_ARCHITECTURE.md) qualified by [corrected decision/support](../reports/post_v5/development-20261009T083921-00e775cf/CORRECTED_RESEARCH_DECISION.md); no final artifact trained |

A separate Prospective Research view was added under Validate. Forecast and Insights forecast requests now archive actual inputs and outputs in the new namespace. Existing V5 evidence/comparison remain intact. All forecasts retain research status, source/health/OOD/provenance and compulsory abstention; buying questions return measured indicator context without an actionable claim. Actual inputs and prospective forecasts remain separate from reconstructed historical studies. Future final and boundary-overlap outcomes remain hidden from research views.

## Current live evidence

The08 October capture preserved100 equity/19 reference payloads, but all100 shadow attempts failed critical price-quality checks. Those failures remain recorded. A fresh09 October capture produced388 forecasts across97 members (four horizons each); ITC.NS, TMPV.NS and VEDL.NS remain with explicit source/session/corporate-action failures. No member was removed. All 388 forecasts are LOW/abstained and use the frozen V5 candidate. Their actual issuance times09:11:03–09:26:23UTC precede the standard Indian session close10:00UTC; inputs were already archived. At closeout verification, zero matured outcomes were attached.

This is genuine new acquisition/forecast evidence, not proof of model skill, historical training PIT safety or official calendar accuracy. Returns/alphas/coverage attach only after exact independently observed sessions mature and each required source was itself acquired after completion. Missing prices/volume/benchmarks stay pending; outcomes do not shift to later candles or become irreversible fake zeros. Price/action-vintage differences are retained.

Explicit UI/CLI collection is active. No recurring OS task was installed; the optional scheduling preference is unanswered. [Operations](POST_V5_OPERATIONS.md) provides the daily runner, private storage and timing instructions. Intraday runs predict from a last-completed-session origin; their remaining lead is shorter than a full session and must be evaluated separately from pre-open issuance.

## Corrected historical findings

Study development-20261009T083921-00e775cf executed480 configurations (four horizons/four dated folds/all100 members attempted), with97 historical eligible securities overall. All transforms and label endpoints are before2025-10-01. Independent source-mask verification checks39,992 scored stock-origin/horizon rows, exact returns/cutoffs and no bad past/forward sessions. Data remain retrospectively revised and current-membership biased.

| Sessions | Eligible validation rows | Zero log-return MAE | Ridge MAE |
|---:|---:|---:|---:|
| 1 | 10,774 | 0.014780 | 0.016288 |
| 5 | 10,386 | 0.034626 | 0.050208 |
| 10 | 9,901 | 0.047734 | 0.079967 |
| 20 | 8,931 | 0.065607 | 0.082113 |

Zero is strongest among the registered simple references at every horizon. No complex model passes the aggregate/fold/60%-stock improvement screen. A reduced1-session model improves MAE only0.126%, within the registered1% tie, and improves53.61%ofstocks. Decomposition adds no demonstrated benefit on matched supported cohorts. Normalized/excess/rank targets do not establish better absolute-return forecasting; their metric spaces remain separate. Sector-specific and interaction fits remain diagnostic.

Source quality is a major limitation:95 of 97 pre-boundary stock histories contain a zero/invalid18March2025 session. The fixed200-session guard leaves quarter3 with only1 stock/1 industry and quarter4 at most2 stocks/2 industries. Those gaps are retained and never filled to increase coverage. Overall97-stock counts therefore do not justify broad four-quarter stability conclusions. Statistical signal weakness and data insufficiency are separate findings.

Raw classifier discrimination is unstable; a5-session aggregate AUC0.557/balanced accuracy0.521 narrowly passes scalar thresholds but has no meaningful within-date ranking, sparse later-quarter support and Brier worse than0.25. This does not justify downstream calibration. Numerical IC artifacts in near-constant predictions are marked undefined in an additive audit, not promoted as stock signal. Effective nonoverlapping date support is much smaller than pooled row counts.

The native V5 replay exactly matches saved fit arrays and original development-validation metrics. Several heads show train/validation degradation, notably India 20-session MAE +91.2%. Released negative-slope Platt calibrators can reverse in-sample ranking; the separate raw-versus-issued diagnostic clarifies that effect. These are measured diagnostic associations, not proof of a single causal failure. No calibrator/model/probability/threshold was changed. Architecture decisions use the corrected development study, not consumed final failures.

A universal-versus-market-specific distinction is not separately identifiable in this India-only study: the pooled model is already market-specific. A multi-market test would need its own broad registered universe and separation gates; no cross-market superiority is claimed.

## Review corrections and immutable history

The first completed post-V5 study was invalidated after review reproduced naive/aware pandas alignment disabling session-quality masks. All original metrics/reports and executed source bytes remain. The corrected study has a new ID; no failed artifact was rerendered. Initial interrupted studies and a failed source-audit assumption about weekend information cutoff versus maturity session also remain, with explicit review/audit corrections. [Evidence index](../reports/post_v5/README.md) prevents old canonical reports being mistaken for current selection evidence.

Independent review found and verified corrections for stale ticker tails, expired cached captures, per-source outcome completion, non-traded/missing outcome endpoints and invalidated-study/UI audit visibility. Focused synthetic fixtures and real native-model integration use only pytest temporary ledgers. No synthetic forecast was inserted into the actual live study.

## Verification and remaining gates

Final full regression: 165 passed, eight retained legacy UTC deprecation warnings,27.72 seconds. Existing/new pages and the real frozen-model shadow/immutable-ledger path pass. App health returned ok; browser pixel/responsive/accessibility QA remains unavailable and unclaimed. Study 39 metric/addendum hashes and 11 source hashes were independently checked; original V5 preservation remains intact. Compiler and final ledger/source checks are recorded in reports/post_v5 verification evidence.

Remaining work is prospective and evidence gated: accumulate real capture/forecast/maturity history; improve legitimate source/calendar quality; inspect rolling/stock/sector/regime/OOD stability; use a new future untouched period for any successor; establish discrimination before calibration; and retain all failures. No NLP expansion, relaxed abstention, final successor fit or production promotion follows this delivery. The evidence supports software research and diagnostics, not reliable trading decisions or guaranteed prices/profit.
