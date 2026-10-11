# Web platform implementation plan
For agentic workers: use subagent-driven-development with one implementer at a time, review each substantive task, and track .superpowers/sdd/web-platform/progress.md.

Goal: deliver the user-approved local/manual web frontend while deferring unapproved cloud/GitHub activation.
Architecture/constraints: WEB_PLATFORM_DESIGN.md is authoritative. Frozen44121a6 research, model/threshold/calibration/feature/abstention/final results unchanged. No GitHub calls or drafts; no remote push/deploy. Existing CLI/local scheduler and Streamlit preserved. Python financial/ML calculations remain authoritative; typed OpenAPI→TS.

## Task1 — Backend/operator readiness
Files: new application/services/web_analytics.py, archived_forecast.py, historical_evidence.py, operator_jobs.py; new application/web_schemas.py; api/web_routes.py, operations_routes.py; minimal api/main.py integration; tests/test_web_services.py, test_operator_jobs.py, test_web_api.py. Do not edit protected core/models/research or original reports.
1. Write failing fixture tests for window enforcement, operator auth/Origin/CSRF, enqueue read-onlyGET/no training, run records and CLI reuse; scenario source immutability, analogue maturity, V5failedgate, financial date/currency alignment and output null/error semantics.
2. Implement shared pure/adapted services using exact existing exports; bounded provider calls and inputs. No user files served. Source/window/model/protocol provenance stays explicit.
3. Return machine job records: runid, scheduled/started/completedUTC, type, protocol, expectedorigin, universe100, successes/failures/abstentions/duplicates/provider/data failures; final SUCCESS/SUCCESS_WITH_DEGRADATION/SKIPPED_NON_SESSION/FAILED.
4. Preserve schemas already consumed; add exchange session_date to bars compatibly and paginated latest/range selection. Add typed detail, hypotheticals, historicalgate, market context, financial tools and protectedstateless inputs.
5. Run new tests +originalsuite, export OpenAPI, preservation checker. Commit local task; review before frontend.

## Task2 — Frontend foundation and shell
Files frontend/package.json/lock/config, generated/types, api/client, query providers, app/layout/globals, sharedshell/search/status/table/loading/error/chart boundaries; coreui tests.
1. Inspect pinned skill context/craft-floor and design tokens. Pin actual npm versions; use AppRouter/TSstrict/Tailwind/Query/LWC/Recharts; no extra UIframework necessary.
2. Generate API types from docs/api/openapi.json; one client, clear errors, URL/env controls, secrets never public.
3. Build sharedshell, semanticdark/light tokens, contextualURL navigation, accessiblemobile drawer/search debounce/recentselection.
4. Testshell/search/researchbadge/abstention/pending/error/loading; typecheck/build skeleton.

## Task3 — Dashboard/Forecast/Validate
Files frontend/app/page, forecast/[ticker], validate and feature components; tests contracts/workspaces.
1. Backend-authoritative dashboard streams/market/state/savedforecast summaries.
2. Saved Forecast4h cards +financialchart +overview/drivers/scenarios/analogues/evidence/historytabs; allgroup/provenance/calibration/null/OOD/abstention semantics.
3. Failed V5gate remains prominently visible; prospective groups independent, tiny samples marked.
4. FixtureUI/contracttests +realreadonlyAPI integration; no job frompageGET.

## Task4 — Research/tools/operator interface
Routes stocks/[ticker],markets,portfolio,watchlist,screener,compare,risk/[ticker],admin/research-status.
1. Sharedcomponents withtypedbackend data and financialcalculations. No JSreimplementation.
2. Guesttracking browserstate and protectedPythonanalysis; no insecure legacyaccountmigration.
3. Operator authentication andmanual jobs withinprotocolwindows, explicitPCdependency/status/failure/time; no trainingbutton.
4. Test criticaltoolinputs and operatorstates; fullbuild/typecheck.

## Task5 — QA/docs/delivery
1. Inspect allmajorpages inbatched desktop/mobile light/dark and1440/1280/1024/768/390 whenbrowseravailable. Accessibility keyboard/focus/forms/chart summaries andno pageoverflow.
2. Measure bundle/routes/chart loading/APIduplication/interactions. Fix materialfindings once, reverify.
3. Independentvisual/code review. No unsupportedbrowser/E2Eclaims.
4. FRONTEND_ARCHITECTURE,API_CONTRACT,UI_DESIGN_SYSTEM,STREAMLIT_MIGRATION,FRONTEND_DEPLOYMENT,AUTH_MIGRATION,WEB_PLATFORM_COMPLETION_REPORT,masterboard withdeferredcloudclearlynotdone.
5. FullPython+frontendtests, build,health,source/model/predictionpreservation,localcommits/cleanstate. Stop; do not deploy/push/train.
