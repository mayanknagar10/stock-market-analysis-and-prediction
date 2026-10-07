# StockPro V5 implementation status

Updated: 07 October 2026. Branch: v5-development. Source of truth: docs/specs/StockPro_Analytics_V5_Codex_Implementation_Spec_v1.1.md and user's supplied phased instructions. Detailed existing architecture, defects, leakage/data risks and reusable modules: REPOSITORY_AUDIT.md. Actual provider inventory: DATA_SOURCES.md and data_sources.json. V4 freeze: reports/baseline/V4_FREEZE.json.

## Proposed architecture and execution decisions

Use additive, independently tested modules under core/data, core/validation and core/forecasting; retain original V4 engine for frozen baseline only. Data adapters -> versioned snapshots + availability checks -> causal technical/context features -> direct market-specific XGB/LGB heads -> disjoint calibration, quantiles, OOD/abstention -> immutable ledger. Training/evaluation remain explicit CLI jobs. UI switches only after those contracts pass. Existing analytics/auth/portfolio are preserved.

A wholesale rewrite would increase regression risk; modifying the V4 module in place would destroy baseline comparability. The additive approach allows recovery, schema checks and evidence gates. Market-specific India/US families have independent artifacts; crypto/indices are unsupported for equity forecasts. Raw prices are display-only; corporate-action-adjusted closes construct labels.

The user phase numbering (0–12) governs execution; spec 10's shorter phase table is mapped into it. Quantile/direction heads can share target/split infrastructure but their acceptance tests are recorded under phase 4. Optional 60D is deferred as specified, with separate validation gate. No deep-learning, paid APIs, external LLMs or unreliable scraping. No UI reliability claims without persisted evidence.

## Ordered phases and acceptance checks

| Phase | State | Deliverables / files | Required tests and exit evidence |
|---|---|---|---|
| 0 Audit + V4 baseline | COMPLETE | Audit/source inventory; freeze/hash V4; baseline metrics/report CLI; existing-behavior tests; timestamp/model contracts | Causal/scale invariant technical features, risk, screen strategy preservation, exact V4 forecast fixture, provenance for baseline dataset; real/synthetic results separated; Streamlit/page smoke |
| 1 Leakage-safe validation | IN PROGRESS | core/validation targets/splits/availability; frozen protocol | Direct labels; purged boundaries including pooled same-date rows; timezone/session joins; preprocessing train-only; final-test exclusion |
| 2 Direct horizons | NOT STARTED | core/forecasting heads/training/inference; scripts/train_v5.py | Independent 1/5/10/20 targets; both XGB/LGB; market family/schema reject; explicit training only; no V4 multi-day calls |
| 3 Context features/data layer | NOT STARTED | core/data adapters/snapshots/quality; contextual feature builder/mappings | Source metadata; stale/failed providers; compatible fallbacks; snapshots replay; US/India sessions; sector mapping; causal relatives; ablations |
| 4 Direction + quantiles | NOT STARTED | Per-horizon classification and native quantile heads | Bounded output; q10/q25/q50/q75/q90 order; pinball loss/coverage/width; head disagreement recorded |
| 5 Calibration/confidence/OOD | NOT STARTED | Disjoint Platt calibration; evidence/abstention policy | Base/calibration separation; Brier/log loss/reliability support; OOD and stale data force abstention; thresholds validation-only |
| 6 Regime/relative forecasts | NOT STARTED | Regime layer, market/sector auxiliary alpha | Regime causal inputs; by-regime OOS; missing benchmark/sector explicit; no arbitrary decomposition |
| 7 Provenance/ledger | NOT STARTED | Append-only SQLite forecasts/outcome events; resolver/analytics | Duplicate/rewrite rejected; complete metadata; immutable outcomes; session maturity; split/dividend resolution; no future outcome joins |
| 8 Scenarios/invalidation/analogues | NOT STARTED | Same model inference override engine; train-only scaled neighbours | Scenario overrides permitted schema/OOD; no invented effects; outcomes known by as-of; sensitivity reruns exact head |
| 9 News/events | NOT STARTED | Normalize timestamped news; event interfaces; optional local NLP | No headline BUY/SELL; availability/first-seen; provider fail-safe; no scraping; gated on proven core |
| 10 Registry/hardening | NOT STARTED | Native artifact manifests; champion/candidate gate; deployment docs | Schema/checksum/version/cutoff; rollback/no overwrite; immutable manifest; explicit training; failure/reload behavior |
| 11 UI redesign | NOT STARTED | Dashboard/Research/Forecast/Validate/Risk & Portfolio/Markets | Existing page mock smoke; truthful distribution/evidence/provenance first; formatting/a11y; no unsupported trust labels |
| 12 Final evaluation/docs | NOT STARTED | reports/V4_VS_V5.md; completion/README/architecture/deployment | Frozen final test once after decisions locked; no superiority claim absent evidence; all critical tests and startup/page checks |

## Migration risks

Preserve models/universal_* files, legacy analytics and local user data. JSON auth writes are concurrency-prone; any storage migration needs backup and rollback. Ephemeral Streamlit hosting loses users/snapshots/ledger unless a durable volume is supplied. Native boosters avoid loading arbitrary pickle artifacts; validate schemas and file integrity before inference. New exchange-session policies may differ from legacy weekday dates; surface the change. Present-day macro/news/fundamentals must remain research-only without true historical vintages. Current-universe survivorship bias must stay in model cards.

## Phase record

Phase 0 audit inventory and frozen-checkpoint diagnostic COMPLETE (20 source/call families; 72 real-equity diagnostic observations per horizon). 14 metric/preservation/loader tests and 13 page smoke checks pass. Market/sector baseline extension and true historical checkpoint lineage remain unresolved; no claim of OOS V4 accuracy. Initially missing runtime dependencies installed into ignored .deps, leaving global Python unchanged. Standard exec helper fails setup; approved execution works. Yahoo direct HTTP endpoint returned 429, while yfinance history route returned 1241 RELIANCE.NS rows. Source failure must not be mistaken for missing market data. No production model/UI architecture changes yet. The Windows LightGBM text file is CRLF but declares LF byte offsets; diagnostic uses an explicitly normalized in-memory copy with integrity checks. Original files remain unchanged. No OOS superiority claim.

## Next testable work

Write baseline metric and preservation tests first; execute them against frozen source; persist evidence. Freeze final-test start globally before loading evaluation rows; retain post-boundary prices sealed from development. Capture fetched/raw provider payloads and hashes. If external data cannot be fetched for a required family, finish independent contracts/tests and document that family as blocked.

Phase 0 exit: 27 automated checks passed, Streamlit health endpoint returned ok with --global.developmentMode=false. No original forecast/UI files changed; all checkpoint bytes unchanged. Per-head unavailable V4 statistics are null. Diagnostic evidence is not historical OOS evidence. Phase 1 can add independent safety infrastructure; promotion still requires reconstructible as-of data and legitimate historical fits.
