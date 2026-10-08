# Model artifacts

The repository contains preserved V4 artifacts and versioned V5 research checkpoints. It is not an empty model directory.

## V5 research registry

v5/ACTIVE_RESEARCH.json points to research-20261008T190859, with separate India/US equity bundles for 1/5/10/20 sessions. Each bundle contains native XGBoost JSON, LightGBM text, schema/scaler/metadata and checksummed auxiliary artifacts. Loading verifies the complete expected file set, family, horizon, schema and data cutoffs. Existing directories are never overwritten. Ordinary forecast requests never train.

**The frozen final performance gate failed. Production inference is disabled.** All current forecasts remain LOW and abstain. Historical input availability/revision vintages and exchange calendars are unverified. These are research artifacts, not evidence of reliable investment decisions or support for every ticker.

v5/QUARANTINE.json identifies rejected candidates; earlier artifacts and development reports are retained. Do not delete failed versions or point the active registry at a quarantined/incomplete bundle. The current experiment covers six equities and limited current sector mappings. Crypto/index/FX families and 60-session horizons are unsupported.

Training is an explicit job, scripts/train_v5.py, using private captured stock/reference manifests. Further modeling requires a separately registered untouched future holdout; the released final period has already been consumed. See [deployment guide](../docs/V5_DEPLOYMENT.md) and [comparison](../reports/V4_VS_V5.md).

## Original V4 freeze

universal_xgb.json, universal_lgb.txt, universal_scaler.json and universal_meta.json are the preserved original checkpoint. Its metadata lists synthetic SYN_0–SYN_14 training assets. Reported synthetic metrics cannot establish real-market historical OOS accuracy. The diagnostic adapter explicitly normalizes the LightGBM text in memory to handle its original Windows line endings; original files remain unchanged.

Do not overwrite these files with the legacy training script when reproducing the V4 freeze. V5 inference uses direct horizon heads and never silently falls back to the universal model. [Original diagnostic](../reports/baseline/V4_BASELINE.md) and [freeze hashes](../reports/baseline/V4_FREEZE.json) document the baseline.

Frozen source/model/config/result files use -text .gitattributes to preserve their exact hashed bytes across Git checkouts. Store private users, snapshots and prediction ledger on durable private storage; they do not belong in this model directory or Git.
