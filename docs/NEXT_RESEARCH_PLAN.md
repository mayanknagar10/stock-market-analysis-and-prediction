# Post-V5 Research Implementation Plan

> For agentic workers: use subagent-driven-development for independent research/report tasks and inline execution for the tightly coupled capture foundation. Track status in docs/NEXT_RESEARCH_TASK_BOARD.md.

Goal: preserve V5 and establish genuine prospective capture plus development-only failure diagnosis.
Architecture: immutable original V5; new research/post_v5 modules; private data/research/prospective-india-20261008-v1 run snapshots/ledger; public config/research and reports/post_v5 reports. Frozen V5 imports are read-only.
Tech stack: Python/Streamlit/pandas/numpy/sklearn/XGB/LGB; existing Yahoo provider.

## Global constraints

Never overwrite V5 reports/models/locks/markers/snapshots/quarantines/ledger. Never read consumed final CSV for decisions. Feature and label development cutoffs are strictly before 2025-10-01. No new final candidate, production claim, BUY/SELL or NLP expansion. Future protocol is docs/NEXT_RESEARCH_PROTOCOL.md.

### Task 1: preservation and protocol

Create scripts/post_v5_preserve.py and reports/post_v5/V5_PRESERVATION.json. Hash protected original directories and ledgers. Test corruption rejection using a temp fixture; use an explicit verify operation that performs no writes. Freeze protocol JSON/membership with exclusive file creation. Record baseline regression results.

### Task 2: capture and universe

Create research/post_v5/archive.py, universe.py and scripts/post_v5_capture.py. Archive source payloads and metadata before inference, store immutable hashed run envelopes. Tests: changed historical revision yields a new ID; modified bytes reject; source fetched after cutoff rejects; sector failures stay in run; frozen member cannot be removed. Download exact official constituent CSV and keep all source members with ISIN/industry and acquisition identity. Capture all members and existing contextual sources, preserving partial/missing failure evidence.

### Task 3: shadow and maturity

Create research/post_v5/shadow.py, scripts/post_v5_shadow.py and post_v5_resolve.py. Read V5 bundles without changing pointer. Use separate PredictionLedger path. Build bounded complete-session features and archive all inputs/features before atomic append. Tests: future input rejected; missing price blocks; frozen model remains LOW/abstained; append duplicate/outcome rewrite fails; benchmark and sector alignment on independently captured maturity dates. Route new explicit UI requests through this namespace; do not change frozen core.

### Task 4: development research and diagnosis

Create research/post_v5/development.py and experiments.py, scripts/post_v5_research.py, tests/test_post_v5_research.py. Load retrospective data and immediately clip dates before 2025-10-01. Tests with a deliberately enormous forbidden future value prove labels/features/fits are unaffected; purge folds globally; train-only transforms/pruning; rank targets compare contemporaneous eligible universe only. Evaluate registered baselines, targets, feature ablations/stability, pooled/sector/interactions and decomposition on fixed folds. Preserve all attempted results with new IDs. Do not save a final trained candidate. Produce reports/post_v5/V5_FAILURE_ANALYSIS.md, BASELINE_COMPARISON.md, TARGET_FORMULATIONS.md, FEATURE_ABLATIONS.md and NEXT_CANDIDATE_ARCHITECTURE.md with real counts/results and negative findings.

### Task 5: research dashboard and verification

Add pages/research_results.py and a separate Validate navigation entry; keep existing comparison untouched. Show baselines/horizons/universe/stock/sector/rolling/regime/confidence/calibration/intervals/abstention and prospective run/failure records separately from retrospective studies. Tests: no prospective outcomes fabricated; missing study gracefully handled; V5 pages still initialize. Run new contracts, full regression, compile, health and preservation verify. Commit logical phases; no push/merge/deploy.

Execution proceeds under the user's explicit instruction; no approval is needed for ordinary reversible implementation choices. Data/model stopping rules still apply.
