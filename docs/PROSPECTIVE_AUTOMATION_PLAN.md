# Prospective Automation Implementation Plan

> For agentic workers: execute collection beforeAPI; independent API implementation and whole-branch review use subagent-driven-development. Track progress in .superpowers/sdd/automation/progress.md.

Goal: deterministic canonical collection/maturity/validation plus read-focused FastAPI shared services.
Architecture: immutable priorV5/b3a9edc research; new application/collection modules, independent jobs, typed services andapi/v1. Separate canonical ledger/control/batches from legacy388; no model/threshold/feature/training change.
Tech stack: Python3.12, SQLite/content-addressed snapshots, existing nativeXGB/LGB andfinancial core, Pydantic2/FastAPI/Uvicorn.

## Constraints

Never rewrite protected reports/models/snapshots/consumedtest or388forecast payloads. Canonical cut06:30IST D+1; capture06:00–06:30, generation06:30–07:00. Currentmodel staysFROZEN andLOW/abstained. No production or Next.js. No source postcut/retrycutshift. Unsupported calendars/specialtimings failclosed. APIreadpaths returntyped DTOs; no training/capture sideeffects.

## Tasks

1. Seal currentresearch bytes/forecast hashes onnewbranch; write protocol and official2026calendar withsourcehash. Test session/holiday/uncertain/special/nextsession behavior and exactcutoff/awaretime.
2. Write tested deterministic collectionstore andcollector with boundedretry/deadline, immutable source/forecast/outcome/batch, concurrency, per-stockpartialresumption and all100statuses. Tests double-run/no providercalls, interruptedresume, corruptedbatch, missingdata/cutoff, no lateforecast.
3. Reusefrozen calculations in pureforecastengine; shared saved-forecast/ledger/validation services. Canonical andlegacy groups separate. Independent resolver for1/5/10/20, benchmarks/sectors/actions/pending/duplicateoutcomes. Tests own-source completion and exact calendar maturity.
4. Onlyafterprotocol/collectorcontracts pass: implementFastAPI Pydanticroute/errorcontracts, shared stock/market/technicals/risk/portfolio/screener/factor services, cache and instrumentation. All required20routes readfocused; auth/CORS configuration andpath/privatepayload redaction. TestClientrealtime and fakeproviderboundary coverage; generateOpenAPI.
5. StreamlitForecast/Insights retrieve savedcanonicalforecasts; researchview uses shared health/aggregation; no pageopenscollection. Providecron/external/local/self-hostedGH scheduling templates andexactenvcommands, deployment/API docs. No hostingdecision orautomatic modeltraining.
6. Verifyfullregression, compile, service/HTTPhealth, OpenAPIcontracts, latencymedian/p95 withfixture andboundedliveprovider measurements, preservebaseline388/source/model hashes. Independentreview/fixcriticalfindings, commitlogicalphases, updateboard/completionreport; stopbeforefrontendmigration.

Initialaudit: existingrandomUUID/variable-requestcutoff, Streamlit-linked forecastrequests, no canonicalbatchcontrol oridempotency. Existingappend-onlyledger/sourcearchive/native model/inference/financial calculations are reusable. Calendarproxy andephemeral hosting require explicitnewcalendar/storage boundaries. SQLite choosesone durablewriter deployment initially; remote multiwriter database migration is separate.
