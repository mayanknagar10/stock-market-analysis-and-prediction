# Web platform design
Baseline44121a6; branch platform/web-migration-20261009.

## Confirmed execution mode
The user explicitly selected the existing local Python backend/storage, protected manual frontend buttons, and deferral of laptop-independent collection until GitHub work is separately approved. The existing CLI and automatic local scheduler stay usable. No GitHub access, uploads, secrets, pushes, repositories, workflow drafts or activation are authorized. The system must remain $0; no paid service or deployment is enabled.

Local/manual readiness replaces the initial cloud-verification prerequisite for frontend development. Cloud independence remains DEFERRED, never described as complete. PC/backend availability is visible in the operator interface. PERSISTENCE_AUDIT.md records the unchanged timing and storage facts.

## Architecture
Next.js App Router + React + strict TypeScript + Tailwind → typed single API client + TanStack Query → FastAPI shared Python services → frozen native models/registry/data/ledger. Generated OpenAPI types are authoritative. React renders/formats; financial, indicator, risk, forecast, SHAP, calibration and validation calculations remain Python.

Extend existing application services rather than protected core/models/research. Add typed current-vintage fundamentals/news, archived forecast detail/scenarios/analogues, historical failed V5 evidence, richer markets/technical series/risk/factors, stateless compare/portfolio/watchlist/screen calculations. No request trains, recalibrates or issues forecasts. Operator job POSTs enqueue the unchanged explicit CLI commands with bounded execution, allowlisted job types and immutable execution records.

## Operator/auth boundary
Admin/Research Status is separate navigation. Job submission requires authenticated operator access. Local operator sessions may be provisioned only with explicit development enablement, loopback client and exact permitted Origin; otherwise a configured operator bearer token is required. Cookies are HttpOnly/SameSite and secrets never use NEXT_PUBLIC variables. Normal GETs remain read-only. No data/users.json route exists.

Guest portfolio/watchlist inputs live in the user's browser; Python performs their calculations. Sensitive analysis/job endpoints are protected; account identity and old Streamlit credentials are not silently migrated. Full hosted auth is a controlled later step, documented in AUTH_MIGRATION.md. No broker execution.

## Visual direction
Dense chart-led research terminal with charcoal/slate surfaces and restrained blue selection. Flat adjoining panels, thin borders, square workspace surfaces, compact controls, system sans and tabular numbers define the implemented interface. Exact tokens and page patterns are documented in frontend/DESIGN.md. Positive/negative encode measured values; amber research/warning states are textual/icon-labeled. No gradients, glow, fake gauges, oversized BUY/SELL or decorative motion.

Desktop:176px left navigation, 50px command/search bar, compact context bar, flexible chart-led workspace and contextual readout rail. Tablet/mobile navigation becomes an accessible drawer; chart/workspaces stack. Light mode uses the same semantic CSS-variable tokens with equal contrast/functionality. Root page is market/research dashboard, not engineering task board.

Forecast leads with company/ticker/current session bar and timestamp plus distinct saved forecast origin. Four horizon distributions preserve LOW/ABSTAIN and recommendation disallowed. Quantile intervals/probability exist only when backend provides them. No frontend interpretation as BUY/SELL. Drivers are actual contributions, not causal claims; scenarios are archived conditional estimates; analogues include matured labels only.

Validate prominently retains FAILED V5 gate. Retrospective/reconstructed, legacy/manual prospective and canonical prospective are separately labeled, with counts, pending/sealed/invalid states and backend metrics. Evidence collection is not an inevitable production pass.

## Shared components/state
ApplicationShell, GlobalTickerSearch, ResearchBadge, StatusLabel, DataModelHealth, EvidenceStatus, Metric, DataTable, EmptyState, ApiError, Skeleton, PriceChart and AnalyticalChart. Query keys include ticker/group/range/protocol-sensitive context; requests deduplicate and only fetch active page datasets. Search debounces300ms, supports arrows/Enter/Escape and recent browser selections.

Candlestick+volume use Lightweight Charts; analytical distributions/metrics use Recharts, lazily loaded. Daily chart dates come from explicit backend exchange-session labels. Client performs no normalization, indicator, portfolio or validation math.

## Routes
Dashboard; stocks/[ticker] with overview/technicals/fundamentals/factors; forecast/[ticker]; validate; portfolio; watchlist; markets; screener; compare; risk/[ticker]; admin/research-status. URLs preserve ticker context. Every route has loading/empty/error/degraded states and retry controls. Historical provenance remains visible.

## Verification
Meaningful backend tests precede new services/operator boundaries; existing203tests maintained. Generate types from exported schema, test consumed structures and critical UI invariants. Test research/abstention never renders BUY/SELL, pending versus realized records, data failure, health, search/mobile navigation and loading.

Build/typecheck/lint/unit tests, backend health and E2E when browser runtime supports it. Responsive widths1440/1280/1024/768/390; dark/light, keyboard/focus, zoom, charts, controlled table scroll, no whole-page mobile overflow. If browser unavailable, report that explicitly. Prepare configurable API/Vercel instructions; no deployment occurs without permission.

## Pinned terminal benchmark — 10 October 2026

The user explicitly pins information density, financial-terminal character, tool discoverability, compact tables/data rows, strong charts, subtle borders and restrained surfaces. Simple/Advanced disclosure and one coherent stock workspace improve discoverability. No oversized rounded-card admin-dashboard grid.

Tradytics public [homepage](https://tradytics.com/) and [charts/tool view](https://tradytics.com/charts) were reviewed. The public chart shell was captured in a fresh headless browser without signup or account actions; full authenticated/premium behavior was not inspected. Grouped chart controls, persistent tool navigation and contextual watchlist/quick-view panels inform the craft benchmark. StockPro retains its own identity, original layout and truthful research semantics. No competitor source code, branding, paid-data capability or recommendation claim is adopted.
