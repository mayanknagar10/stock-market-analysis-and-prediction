# Independent prospective automation review

Date: 2026-10-09. Branch: `research/prospective-automation`. Baseline: `b3a9edc`.

**Accepted within the reviewed scope: no unresolved Critical or Important findings.** Review covered the collection protocol and plan, collection/store/calendar/inference/outcomes, shared services and API, Streamlit saved-forecast integration, jobs, tests and deployment documentation. The reviewer changed only review reports.

## Resolved findings

- Insights now retrieves saved canonical forecasts through the application service without invoking the legacy forecast writer.
- Required stale context is rejected before inference; optional unused sources retain explicit degraded status.
- Issuance records and batches retain code, frozen model artifact/parameter, manifest and feature-vector identities. Frozen context configuration is included; integrity failures report FAILED even after partial completion.
- Finalization rechecks the deadline after inference and before ledger admission.
- Probability metrics score raw probabilities under the raw diagnostic label; frozen calibration remains unchanged.
- Provider attempts are durably recorded before the external call and consume the bounded retry budget across crashes.
- Provider cache capacity is bounded; history/search items have typed contracts.
- Outcome resolution uses post-capture/per-row clocks, rejects future receipts including every contributing fixed peer, and retains canonical/legacy separation.

## Independent verification

- Actual preserved legacy ABB.NS workspace loaded four horizons with provider capture prohibited. Zero-shock scenario returns matched all four saved central log returns exactly.
- Final-window fixture masked both an origin inside the final window and a pre-final origin maturing inside it, while retaining an earlier mature outcome.
- Automation collection/API/assistant/outcome/scheduler run: **34 passed in 41.64s**.
- Updated outcome/API run: **17 passed in 13.80s**.
- Final peer receipt, validation state and immutable all-member batch checks: **3 passed in 4.14s**.
- Fixed-universe failures remain represented in all-100 batch statuses. Forecast/validation contracts enforce research-only, LOW/abstained estimates, recommendation disallowed and production promotion blocked.

## Initial outcome timestamp amendments

The initial resolver wrote 77 outcomes with a resolution timestamp earlier than source acquisition. Original prediction and outcome payloads were preserved. Hash-bound append-only annotations record a later audited acceptance time; they do not claim the original execution occurred at that later time. Unamended invalid clock ordering is excluded from metrics.

The reviewer independently checked all 77 annotation-to-original hashes and recomputed stock/benchmark returns, one direct-sector return and 68 fixed-peer sector returns from 100 captured snapshots: **zero discrepancies**. Every checked source receipt preceded its audited effective acceptance. At verification, the legacy service exposed **388 forecasts, 97 matured outcomes and zero invalid outcomes**, separately from canonical evidence.

Full regression, final preservation verification, measured performance, scheduler installation and deployment checks are recorded separately by the implementing agent. This review does not imply a successful live canonical collection window, sufficient prospective evidence, or production eligibility.
