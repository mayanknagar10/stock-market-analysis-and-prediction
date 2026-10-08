# V5 completion report

Date: 08 October 2026. Branch: v5-development. Research candidate: research-20261008T190859.

**Research implementation and final documentation are complete. The full production definition of success is NOT met.** The untouched numerical research gate failed; historical point-in-time certification is unavailable; production inference remains disabled. This is a delivered experimental research system with explicit blockers, not a validated investment product.

## Scope and authorization

The user's pasted request governs the work; the supplied v1.1 specification defines its contracts. The user confirmed there are no historical availability/revision archives or historical checkpoints, then authorized proceeding with the best research-only implementation. That scope adjustment is recorded rather than weakening certification. Phases were implemented additively on a recoverable development branch. No push, merge, deployment or destructive migration was performed.

## Changes delivered

- Audited the original repository and 20 source/call families, froze V4 source/artifact hashes, and persisted real-equity diagnostics separately from synthetic training metrics.
- Centralized provider access, added immutable snapshots, source/availability/vintage/adjustment contracts, integrity checks, structural quality reports and fail-safe event interfaces.
- Added direct India/US 1/5/10/20-session regression, binary direction and five conditional quantile heads, development ablations and market/sector excess-return heads where data permits.
- Added disjoint blend/calibration/validation blocks, rejected harmful probability calibrations, uncertainty coverage metrics, OOD, evidence thresholds and compulsory research abstention.
- Added native registry manifests, complete artifact checksums, family/schema/cutoff enforcement and preserved quarantined candidate history.
- Added an atomic append-only prediction ledger, explicit matured-outcome resolution, native SHAP associations, source-based scenarios, sensitivity/invalidation and already-matured historical analogues.
- Added Forecast/Validate/Dashboard, consolidated navigation and calm shared UI tokens while retaining existing analysis, risk, portfolio, strategy, factor, auth and personalization tools. Dashboard displays the single master task board.
- Persisted the one-shot final comparison and updated architecture, deployment, source, README and phase documentation.

## Final architecture

Data adapters -> dated snapshot store/quality/availability checks -> technical and selected context feature groups -> independent market/horizon return, classifier and quantile heads -> disjoint calibration/OOD/evidence/abstention -> immutable ledger -> Forecast/Validate interfaces. Training, capture and outcome resolution are explicit jobs; requests never train or silently substitute a legacy predictor. [Architecture details](V5_ARCHITECTURE.md).

Native XGB JSON and LGB text artifacts avoid arbitrary pickle loading. Model cards include parameters, source and feature hashes, exact schemas, training universe, creation time, all data cutoffs, calibration acceptance and validation evidence. Every active artifact shares the training job's frozen source hash. Earlier failed candidates and their reports remain visible; no production champion is created.

Yahoo raw OHLCV, adjusted total-return closes, dividends/splits, exchange timezone, acquisition time and provider version are retained. Reference failures are explicit. Raw snapshots are private/ignored. Holiday placeholders are quarantined only when independent positive-volume session evidence supports exclusion; unresolved expected-session rows block affected features/labels. Structural validity does not certify historical source availability.

## Validation and leakage protections

The protocol boundary was fixed at 2025-10-01 UTC before development. Expanding global-calendar splits purge overlapping forward labels, including pooled same-date assets. Scalers fit training rows only. Feature-group selection occurs before blending. Base fitting, blend weights, calibration and development validation use four separately purged blocks. Calibrators never use base/blend rows. Later acquired reference data are delayed conservatively but remain retrospectively reconstructed and unverified.

Inference checks model creation and all fit/calibration/validation cutoffs; rejects production mode and incompatible schema/family; keeps acquisition timestamps when using cached partial bars. Analogues include only outcomes matured by the request cutoff. Ledger writes are atomic and immutable; maturity uses a known origin and independent session proxy rather than shifting over missing candles. Corporate actions suppress misleading raw-price error.

The final protocol locked exact source and artifact hashes plus preset decision gates. It was consumed once after all modeling decisions were fixed. No final-period tuning or retraining followed. Git normalization is disabled for frozen code, model/config artifacts and validation results so checkout preserves their exact hashed bytes; staged blobs are checked against the lock. The refit V4 architecture uses strictly pre-boundary labels; the original full-history synthetic checkpoint is only a separate diagnostic. This retrospective experiment is not a historically available production model or certified PIT evaluation.

## Final results and calibration

| Frozen gate | Actual | Required | State |
|---|---:|---:|---|
| MAE improvements over zero-return | 1/8 | 6/8 | FAIL |
| RMSE improvements over zero-return | 0/8 | 6/8 | FAIL |
| Brier below 0.25 | 2/8 | 6/8 | FAIL |
| 80% coverage within ten percentage points | 7/8 | 6/8 | PASS |
| Development-accepted calibrated heads | 3/8 | 8/8 | FAIL |

India 10/20-session probability and direction quality deteriorated on the final period. Some nominal 50% intervals undercover, despite acceptable broad 80% coverage in seven groups. Calibration acceptance on development data did not ensure future calibration. All research forecasts are LOW and abstain: actionable coverage 0%, abstention 100%. No HIGH/MEDIUM reliability or forecast superiority claim is warranted.

[The complete comparison](../reports/V4_VS_V5.md) reports all eight pairs, full-grid V4 compounding/naive baselines, genuinely matched recursive V4 origins, classification versus return direction, Brier/log loss/ECE, intervals, relative heads and sector/regime/year/stock stability. Machine JSON/CSV, the lock and consumed marker remain available. Counts are overlapping/dependent; no statistical significance or profit claim is made.

## Verification

The final full regression run returned **122 passed, eight warnings, 33.54 seconds**. Warnings are retained datetime.utcnow calls in original display helpers/market overview; they do not establish timezone correctness and are separately documented debt. Temporal and snapshot contracts use aware timestamps. Functional tests cover all 13 existing page initializations plus Dashboard/Validate and V5 formatting; a separate Forecast route initialized without exceptions. Source/semantic reviews and accepted/rejected forecast fixtures passed. Browser visual/accessibility QA could not run because the browser runtime returned no available browser; pixel, responsive and screen-reader correctness are not claimed.

Final compile, report rendering, persisted-result integrity and startup checks are recorded in [verification evidence](../reports/validation/V5_VERIFICATION.json). There was no configured linter/type-check pipeline to certify. No timing improvement over V4 is claimed: inference stays separate from explicit training, data caching is dated/limited, and checkpoints are cached by version. Preserved native artifact history occupies approximately 136 MB and is retained for audit.

## Unsupported cases and remaining work

| Work | State / dependency |
|---|---|
| Historical availability/revision certification | BLOCKED: actual archives are absent |
| Verified exchange/special-session calendar | BLOCKED for certification: observed peer calendars are proxies |
| Production promotion | BLOCKED by both data eligibility and failed balanced performance gate |
| New NLP/finance sentiment and structured events | DEFERRED under spec §10: first prove the core beats baselines; VADER and fail-safe interfaces remain |
| Browser desktop/mobile/accessibility review | NOT DONE: no browser runtime was available |
| 60-session, crypto, index and FX forecast families | Unsupported; require separate data/model/validation gates |
| Wider universe and historic sector membership | Not validated; current six-stock membership has survivorship/small-sample bias |
| Live production performance | No history exists; ledger supports future recording without fabricated outcomes |

Further debt: free-feed incompleteness, source revisions and calendar uncertainty; conditional intervals under distribution shift; correlated overlapping samples; experimental OOD thresholds; limited sector histories; uncertain snapshot licensing/durability for a deployment; JSON auth concurrency; durable private storage/backups; dashboard filters/analytics for a larger live ledger; automatic benchmark/sector acquisition in the resolver (currently optional and null when unavailable); retained legacy UTC deprecation warnings and Streamlit use_container_width migration warnings. User data were not migrated or published.

## Future work and decision support

The evidence supports inspecting, testing and studying the implementation experimentally. It does **not** support relying on this candidate for real-world trading or financial decision support. Better UI and stronger auditability do not compensate for the failed numerical gate.

Next work should obtain actual historical availability/revision archives and verified calendars, expand the universe with dated memberships, register a new future holdout before any modeling changes, evaluate simple regularized/tree baselines and dependence-aware stability, then reassess probability/interval calibration and live ledger evidence. News/NLP, extra horizons and production promotion should proceed only after their stated prerequisites pass. No future price accuracy or profit is guaranteed.
