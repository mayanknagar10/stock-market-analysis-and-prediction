# Post-V5 research protocol

Protocol ID: prospective-india-20261008-v1. Registered before any new experiments. Branch: research/prospective-20261008. Research only; no production promotion is authorized.

## Preserved V5 and excluded period

Completed V5 commit: 26c09e0. Candidate research-20261008T190859, all model/source artifacts, snapshots, quarantines, ledgers, final locks/markers and reports are protected. A new preservation manifest hashes the original files; subsequent checks reject modifications. The original v5-development branch stays at its completed commit. Every new report/capture/ledger is under a separate post_v5 namespace.

The consumed V5 final period starts 2025-10-01 and extends through the original 2026-10-07 capture maturity cutoff. It can no longer support model selection because its errors and labels are already known. Its existing published aggregate result may be cited as a historical fact or degradation summary; no row-level final predictions or prices from that period may drive any new feature, target, model, threshold or architecture decision. No final V5 report is rerendered or overwritten.

## Development data and fixed experiment budget

Retrospective development uses only observation/feature dates and label endpoints strictly before 2025-10-01 UTC. Data are trimmed before transformations. Current snapshots and current universe membership introduce historical revision and survivorship bias; all such analyses are labeled reconstructed historical research. New public historical capture may expand this development panel but is not prospective historical evidence.

Use fixed globally dated expanding folds: validation 2024-10-01–2024-12-31, 2025-01-01–2025-03-31, 2025-04-01–2025-06-30, and 2025-07-01–2025-09-30. Fit labels must mature strictly before each validation origin block. Preprocessing, correlations, feature pruning, weights and any sector identity interactions fit training only. No random temporal split. Hold out entire stocks/sector groups in supplementary cross-sectional checks. Report sample/session counts and overlapping-label dependence; date-level block summaries are the primary support unit.

Evaluate 1/5/10/20 sessions independently. Pre-register zero, expanding historical mean, scaled 20-session momentum, 5-session mean reversion, ridge, elastic net, shallow gradient boosting, market-only and market-plus-sector baselines. Fixed parameters: ridge alpha 10, elastic net alpha 0.001/l1_ratio 0.5, shallow boosting 50 trees/depth 2/minimum leaf 100/learning rate 0.03, seed 42. No automatic parameter search. Compare broad feature groups, reduced stable sets, pooled/sector/sector-interaction variants, and explicit return decompositions with the same folds/cohorts.

Targets: cumulative log return; ex-ante 20-session volatility-normalized return; market excess; sector excess where references exist; positive return above 0.003 round-trip cost; and same-origin cross-sectional rank. Ranking metrics and normalized errors are not directly compared to absolute-return MAE. Reconstruct return-space estimates before comparing economic errors. No claim that a different target removes noise without measured development stability.

At most two documented specification revisions before prospective candidate freeze. Stop increasing complexity when no model improves zero and the strongest simple baseline consistently across at least three of four folds, at least 60% of eligible stocks, and sector/regime strata with adequate sample support. Prefer a simpler tied model (within 1% aggregate MAE). An architecture proposal is provisional; no next final candidate is fitted until all eight immediate deliverables exist and are reviewed.

## Prospective capture and fixed universe

Freeze the official Nifty 100 constituent CSV as retrieved on the study registration date, including issuer/ISIN, Yahoo .NS symbol, industry, source URL, acquisition timestamp, exact raw membership bytes/hash and the selection rule. Nifty 100 provides broader large-cap sector coverage; it is not a guarantee of individual liquidity or provider availability. Record trailing positive-volume/turnover diagnostics where available, but never drop members because their data or performance is poor. Delisted, suspended and renamed members remain with explicit status and immutable symbol-mapping events. Do not retroactively replace members when the index rebalances.

This solves prospective membership retention from the freeze date onward. It does not solve historical survivorship bias. Historical breadth comparisons using this list remain retrospective and biased.

Starting immediately, explicit UI/CLI shadow forecast runs archive exact provider-normalized raw inputs and source metadata before inference. Each immutable run envelope records observation and real retrieval timestamps, provider/version, OHLCV, adjusted prices/actions, references, sector, FX, commodities, volatility, input/source hashes, feature vector/schema, model/feature version, all fit cutoffs, prediction cutoff, return/probability/quantiles, raw versus calibrated state, confidence, OOD and abstention. Missing sources and failed forecasts are stored, not dropped. Input availability means observed no later than that actual forecast cutoff; it does not certify old historical release/revision times or repair the V5 model's training lineage.

No historical snapshot is rewritten. New revisions create new snapshots. A run cutoff is set after all inputs have been acquired; incomplete current bars are excluded by their original retrieval time and session-completion policy. UI remains RESEARCH MODEL, and frozen V5 is only the shadow reference model. Forecast/output storage uses a new append-only SQLite ledger and a content-addressed run store. V5 pointer/artifact/source files are read-only. All forecasts abstain; no BUY/SELL recommendation is introduced.

Mature horizons through an explicit resolver using independently captured sessions. Attach stock, benchmark and sector returns where available, error, classifier and regressor direction, 50/80% coverage and alpha once. Preserve prediction bytes unchanged. Missing/delisted bars yield pending/unresolvable status, not a shifted maturity date or removal. Corporate-action/revision ambiguity remains flagged. No matured prospective record exists before real time passes.

## Future untouched evaluation

Prospective development/warm-up origins: study freeze through 2027-10-10 UTC. Freeze a next candidate, its complete code/model/data schema and all selection/calibration/abstention rules no later than 2027-10-10, and purge development labels that mature on/after the final boundary. The future untouched origin window is 2027-10-11 through 2028-10-10 UTC, with final scoring only after all required 20-session labels mature (not before 2028-11-10). No outcomes from this final window enter training, selection, calibration or threshold updates. Monitoring may inspect operational source failures and aggregate capture counts only; sealed model-comparison outcomes are opened once by the explicit evaluator after lock verification.

Require at least 252 captured eligible origin sessions per horizon, 80% fixed-universe observation coverage, ten represented industry groups and at least 50 nonoverlapping 20-session date blocks across prospective development plus final/live evidence before promotion review. If counts are insufficient, declare insufficient evidence; extend the observation end only by a pre-registered calendar rule, without peeking at predictive outcomes. Dates never move retrospectively to favor metrics. Until candidate freeze, V5 shadow outcomes are an open warm-up dataset, never called the successor final test.

## Calibration, abstention and promotion

First require development classifier ROC-AUC at least 0.55 and balanced accuracy at least 0.52 on held-out folds, with stable date-level rank information. If absent, stop calibration work and expose raw scores. If present, fit a calibrator on predictions disjoint from base fitting and blending, and require held-out Brier/log loss improvement versus raw scores and training-rate/coin-flip baselines. Report reliability bins, sample counts, ECE and uncertainty rather than asserting calibration from fit alone.

Keep existing abstention safeguards. A successor must independently earn coverage; no weakening thresholds to increase it. Promotion remains blocked unless a separately authorized future decision passes pre-registered prospective gates: at least 2% MAE and RMSE improvement over the strongest preselected simple baseline in at least three of four horizons, no horizon more than 1% worse; positive date-block rank IC with dependence-aware confidence; classifier discrimination/calibration criteria above; 50/80% coverage within five percentage points with useful widths; improvement in at least 60% of stocks and supported sectors/regimes; documented live confidence-bucket and OOD behavior; >=80% source coverage; verified input capture and outcome lineage; sufficient independent date blocks; and operational/visual/data review. Failure or insufficient evidence means no promotion. No guaranteed profit or price accuracy claim is allowed.

## Immediate deliverables and stopping rules

1. Preserve V5 with branch identity and byte-hash manifest.
2. Freeze protocol and prospective membership.
3. Implement and verify new immutable run capture, shadow ledger and maturity jobs; execute an initial capture.
4. Produce V5 failure analysis from pre-boundary development, using final aggregates only as the published failure record.
5. Produce simple-baseline, target-formulation, feature-group/stability/decomposition/cross-sectional development reports.
6. Propose the next architecture using only those development reports; do not train its final artifact.
7. Extend research-results views while preserving the V5 Forecast/Validate record and showing historical versus prospective evidence separately.
8. Verify preservation, tests, data failures, schemas and timestamp/ledger invariants; report pending future evidence honestly.

Stop and report a genuine source blocker if official membership cannot be retrieved; never invent a constituent list. Capture failures do not remove members. Stop an experiment on leakage/schema/corrupt-source detection. Log every attempted configuration, failure and negative result in append-only experiment records; new experiments receive new IDs. NLP/news expansion remains deferred until the market-derived core demonstrates value.

Official universe source: [Nifty 100](https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-100). Existing free Yahoo adapter supplies market data; acquisition archives establish input existence at forecast time, not exchange-grade latency or historical PIT certification.
