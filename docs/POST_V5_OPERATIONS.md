# Post-V5 prospective operations

Study: prospective-india-20261008-v1. V5 remains frozen at commit26c09e0 / model research-20261008T190859. No production promotion or successor final fit is enabled.

## Explicit live cycle

Run from the repository in the existing isolated Python environment:

~~~powershell
python scripts/post_v5_daily.py
~~~

This verifies V5 preservation, captures the fixed100 equities and reference inputs, issues frozen-model shadow forecasts into the new ledger, then attempts to resolve older matured predictions. Every operation receives a new ID. Provider and forecast failures remain visible. It never trains, calibrates, changes a model pointer or overwrites a snapshot.

Individual jobs:

~~~powershell
python scripts/post_v5_capture.py
python scripts/post_v5_shadow.py --run-id NEW_CAPTURE_RUN_ID
python scripts/post_v5_resolve.py --run-id NEW_CAPTURE_RUN_ID
python scripts/post_v5_verify.py --run-id NEW_CAPTURE_RUN_ID
python scripts/post_v5_liquidity.py --run-id NEW_CAPTURE_RUN_ID
python scripts/post_v5_preserve.py
~~~

Use a fresh capture, not an old run replay presented as a new forecast. The shadow service rejects missing ticker tails, expired captures and inputs not archived by the forecast cutoff. Each outcome source is bounded by its own actual retrieval time and completed-session policy. Missing/zero-volume stock endpoints and missing benchmarks stay pending; maturity is never shifted to the next available quote. Sector returns require matching index endpoints or complete fixed-industry peer coverage. Original and later origin prices/actions are compared explicitly.

Prefer a capture before the next Indian trading session opens, after prior global reference markets close. Intraday requests remain research forecasts anchored to the last completed daily close; their remaining lead time is shorter than a full session. Evaluate them separately when judging daily timing. Calendar policies remain research proxies until a verified exchange calendar is added.

Automatic scheduling has not been installed. The current mode is explicit UI/CLI runs; the optional scheduling preference remains unanswered. If scheduling is requested, use the daily runner on an awake, logged-in host with a fixed documented cadence. Missed runs must remain missing; never backfill them as prospective predictions.

## Private storage and replay

Retain data/research/prospective-india-20261008-v1 on durable private storage. Its snapshots, run envelopes, forecasts, attempts, outcomes and shadow_predictions.sqlite are excluded from Git. Keep backups with hashes; do not replace older revisions. Provider-normalized raw OHLCV/actions are archived, including missing/nonfinite fields. This is not a raw HTTP-packet archive or proof of historical publication timing.

Read-only reproducibility requires the exact original private snapshots referenced by the public capture/study manifests. Recapturing changes vintage and creates a new experiment. The active corrected development study points to immutable source backups, metric hashes, source-mask and numerical audits. Historical studies remain reconstructed and survivorship/revision biased.

## Views and release boundaries

Forecast and Insights forecast requests use the isolated prospective namespace. Existing V5 Validate/final comparison remain unchanged. Validate → Prospective Research separates live capture/ledger from historical experiments, shows failed members, coverage, source health, IC audits and the current task board.

Original V5 final data must not be reevaluated or used for decisions. Future candidate freeze is before2027-10-11; untouched origin window2027-10-11–2028-10-10, earliest final scoring2028-11-10 after maturity and evidence requirements. Future-window outcomes and boundary-overlapping warm-up outcomes are withheld from research views. Production remains blocked even after the eight initial research deliverables exist; adequate future evidence and separate authorization are required.

On Windows, deep immutable audit/source backups require Git long-path support. Use git -c core.longpaths=true for clone/checkout/staging, or set core.longpaths=true in this repository only. The research work preserved the original file locations and hashes rather than shortening or replacing audit history.
