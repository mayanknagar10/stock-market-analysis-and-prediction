# V5 research architecture

The implemented application is an experimental research system, not a certified production forecast service. The user authorized research-only development after confirming that historical availability/revision archives and historical checkpoints do not exist. Production inference is explicitly rejected.

## Data and source governance

All active server provider operations reside under core/data. Compatibility modules retain the existing analytics APIs. The V5 Yahoo adapter captures vendor OHLCV with auto_adjust=False, total-return Adj Close, actions, timezone, actual acquisition time and provider version. Historical availability and revision verification remain false. Indices/FX/crypto cannot enter an equity family implicitly. SEC requires an owner-supplied STOCKPRO_SEC_USER_AGENT; no invented contact or credentials are used.

SnapshotStore writes content-addressed UTF-8 JSON with integrity checks, original dtype/index timezone/precision/frequency, and defensive frame copies. Unsupported categorical schemas are rejected explicitly. Historical replay checks publication and revision vintage separately from retrieval time. Corrupt snapshots fail closed; a missing private snapshot on a fresh deployment can be recaptured from the same public provider during an explicit request, with new provenance.

Research uses independent positive-volume peer sessions as a calendar proxy. Flat zero-volume bars absent from that calendar are quarantined, with the original snapshot retained. Zero-volume bars on an expected session remain unresolved; affected feature/label windows are excluded or forecast requests rejected. Standard session cutoffs are timezone-aware; weekend special sessions use a conservative next-UTC-day estimate. None of this substitutes for a verified exchange calendar or historical publication archive.

## Features, labels and evaluation

The original scale-free technical calculations remain in core/indicators.py. V5 adds market/sector/global references, relative strength, rolling beta/correlation and descriptive regimes. Reference series are delayed to the following UTC day for conservative research joins. Every reconstructed availability policy is labeled unverified. Current fundamentals, news, revised World Bank values and Fama-French vintages are excluded from historical model inputs.

Direct labels predict cumulative log returns over 1/5/10/20 observed eligible sessions. Actual origin/maturity session identities are separate from information cutoffs. Global-calendar expanding splits purge training label endpoints touching validation. Preprocessing fits only bounded training rows. The final period starts at 2025-10-01 UTC and is consumed once after source/artifact hashes and decision rules are locked.

Feature-family ablations run only on pre-blending development folds. A group is retained when MAE improves overall and in at least two such folds. XGBoost and LightGBM regression, binary direction and native conditional quantile heads are independent per family/horizon. No deep learning or external LLM is introduced.

The final research fit uses four purged blocks: base fitting, blending, probability/conformal calibration, and development validation. Regression, classifier and quantile blend weights use the blending block. Platt and conformal adjustments use later, disjoint labels. Probability calibration is rejected if validation Brier or log loss worsens. Rejected heads expose raw scores explicitly, not calibrated probabilities.

## Confidence and evidence

OOD measures RMS distance in the training-fitted robust feature space against its empirical 99th percentile. It is a distance warning, not a probability. Uncertainty intervals may expand conservatively for OOD; that expansion has no asserted coverage guarantee. Confidence combines direction/return agreement, calibration support, data eligibility, OOD, interval/noise/cost evidence and learned development edge thresholds. Every research forecast remains LOW and abstains because PIT and production gates have not passed.

Native tree contributions are weighted SHAP associations in log-return units. They are not causal claims. Scenarios alter actual source states, rebuild the same feature pipeline and rerun the same checkpoint; unsupported sources fail explicitly. An abstaining base forecast has no actionable thesis to invalidate. Historical neighbours are selected in training-scaled feature space and include only outcomes already matured at the request cutoff; overlapping neighbours are not independent evidence.

Market/sector auxiliary heads forecast log excess returns where matching reference sessions exist. Calibration state and unavailable sector coverage remain explicit. No arbitrary additive decomposition is presented.

## Registry and ledger

Model artifacts use native XGB JSON/LGB text plus JSON manifests and optional NPZ analogue arrays with allow_pickle=False. Loading verifies manifest/schema hashes and the complete expected artifact set. Existing versions are never overwritten. Quarantined candidates and development reports remain accessible. ACTIVE_RESEARCH is a research pointer; no production champion is created. Registry loading verifies requested family/horizon. Inference checks all fitted/calibration/validation/creation cutoffs and never trains or silently falls back.

Forecast requests validate every horizon before committing the batch atomically to a local SQLite append-only ledger. Forecasts and outcomes have content hashes and update/delete rejection triggers. Full feature values, source snapshots, provider/model/feature/code versions, timestamps, quantiles and trust reasons are retained. Outcomes resolve through a separate explicit job using a known origin and independent maturity calendar; missing candles cannot shift maturity. Splits/dividends suppress ordinary raw-price error when interpretation is not comparable.

## Application and preserved tools

Navigation: Dashboard; Research; Forecast; Validate; Risk & Portfolio; Markets. Existing indicators, risk, portfolio, screener, compare, strategy, factor, auth and personalization functions remain. The rule-based assistant now calls the same V5 research engine. Dashboard reads one master task board. Training and outcome resolution are explicit jobs. Streamlit caches dated data for 300 seconds and loaded models by version; cached partial bars cannot become completed after their original acquisition time.

Known limits: six-stock training universe, current sector membership, unverified historical vintages/calendar, correlated/overlapping samples, unsupported 60D/crypto/index families, incomplete free feeds, gated NLP/event enhancements, and local storage durability. Source/semantic UI and functional tests are available; browser visual/accessibility QA was unavailable in this environment.

Frozen source, native models, protocol config/evaluator and validation results are stored with Git text normalization disabled in .gitattributes. This preserves the exact LF/CRLF bytes required by the protocol and original V4 checksums across checkouts. Frozen byte content was not rewritten; staged Git blobs are verified against the lock.
