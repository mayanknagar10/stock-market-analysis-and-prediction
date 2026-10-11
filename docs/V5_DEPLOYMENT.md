# Local execution and deployment

Tested on Windows with Python 3.12.10. Use an isolated environment; the working session installed dependencies under ignored .deps and did not modify global Python.

~~~powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m streamlit run app.py --global.developmentMode=false
~~~

The app uses saved research checkpoints. Ordinary page loads and forecast requests never retrain. Default sources remain free/public and require no prediction API or LLM credentials. A fresh deployment can collect missing public reference snapshots on an explicit forecast request, while recording new source provenance; failures are visible. Some free feeds are incomplete and forecasts may be rejected.

## Explicit jobs

~~~powershell
python scripts/capture_baseline_data.py
python scripts/capture_context.py
python scripts/train_v5.py --manifest PATH_TO_CAPTURED_STOCK_MANIFEST
python scripts/resolve_predictions.py --as-of 2026-10-08T21:00:00Z
~~~

Use the exact original manifests to reproduce research evidence; recapturing prices changes the data vintage and is a new experiment. Capture/report writers use exclusive-create behavior. Archive private raw snapshots with their IDs rather than adding them to Git. Retain native model directories, manifest hashes and quarantine history for rollback. Set the research pointer only to a complete checked version; never rewrite a saved checkpoint.

For a separately registered future experiment, a final test is locked with evaluate_final.py --freeze-only (with the required --manifest and references), then consumed once by evaluate_final.py. The current release period is already consumed; do not rerun it. It is not a daily development check. The consumed marker prevents reruns. After the period is opened, use a new future holdout for any model changes; do not tune on the reported result.

## Storage and credentials

Persist data/users.json, data/v5/predictions.sqlite and data/v5/snapshots on a durable private volume. Host filesystem durability varies; verify it before deployment. Back up before any user-data migration. Never commit user stores, credentials or private raw snapshots. Existing JSON authentication behavior is preserved; concurrency/durable-auth upgrades remain separate work.

STOCKPRO_SEC_USER_AGENT must contain the deployment owner's actual SEC contact identity before SEC calls are enabled. Optional legacy Polygon/Tiingo keys remain environment/Streamlit-secret hooks; V5 defaults do not require them. No credentials or paid services were introduced.

## Release gate

This release is research-only. Historical availability/revision archives and verified exchange calendars are absent; production inference is disabled even if a numerical result is favorable. The final report records whether the frozen research metrics improved. No result guarantees profit or future prices.

Browser-based visual/accessibility verification was unavailable. Before wider deployment, inspect desktop/mobile layout, keyboard focus, zoom, chart legibility and screen-reader behavior. Functional Streamlit page tests do not establish visual accessibility.

For read-only closeout verification, start a local research instance on port 8515, then run python scripts/verify_v5_release.py. It checks frozen file hashes, consumed result consistency, documentation links, registered routes, compilation and the health endpoint. It reads the recorded test result; it does not rerun the full test suite. python scripts/render_final_report.py only regenerates Markdown from the saved final JSON/CSV. Neither command trains or consumes the final period again.

Frozen source, native models, protocol config/evaluator and validation results are stored with Git text normalization disabled in .gitattributes. This preserves the exact LF/CRLF bytes required by the protocol and original V4 checksums across checkouts. Frozen byte content was not rewritten; staged Git blobs are verified against the lock.
