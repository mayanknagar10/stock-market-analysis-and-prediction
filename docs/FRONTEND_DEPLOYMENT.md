# Frontend deployment

## Separate frontend and backend

The Next.js frontend is prepared for deployment independently of Python. No Vercel deployment, GitHub operation or paid resource is enabled. The current approved operating mode is a local Python backend with private persistent storage. A remote frontend cannot make that PC continuously available.

Runtime: Node.js24 and the pinned npm lockfile under frontend/. Install with npm ci, generate types with npm run generate:api, then run npm run typecheck, npm test, npm run lint and npm run build. The production build uses webpack as pinned in package.json. Build/test results are recorded in WEB_PLATFORM_COMPLETION_REPORT.md after verification; this document alone is not proof of a passing build.

Configure NEXT_PUBLIC_API_BASE_URL to the browser-reachable FastAPI origin (for example https://api.example.invalid). It is a public URL, never a token. The API prefix is /api/v1; client normalization must avoid duplicate prefixes. Configure exact STOCKPRO_CORS_ORIGINS on the backend to the final frontend origin. No production localhost URL is hard-coded. A missing URL must show a configuration error instead of silently connecting to a developer PC.

Development uses the loopback frontend/API with the same hostname, different ports. The user-started background helper runs them without terminal windows; manual CLI commands remain available if the user chooses to use them.

## Vercel preparation

Set the project root to frontend, use the Next.js framework preset and the pinned Node version. Set the API URL for each environment; preview origins also need explicit backend access policy. The frontend requires no writable local filesystem. Guest tracking data live in the browser, not a deployment volume. Do not configure operator tokens or read credentials as NEXT_PUBLIC values.

TLS, allowed origins and authentication must be verified against the actual backend before public sensitive workflows. Public reads can be deployed only according to the intended data-access policy. Shared read-token backends require a separately reviewed server-side proxy/auth integration; do not embed a shared secret in JavaScript. Hosted account authentication remains controlled work in AUTH_MIGRATION.md.

## Backend and scheduler prerequisites

FastAPI requires its own compatible Python runtime, native dependencies, frozen model artifacts and persistent private research data. Local operator provisioning is intentionally development-only and loopback-only. A remote deployment requires separate operator authorization. Scheduled research jobs require durable storage, correct clocks, the registered calendar and one writer. These are not supplied by the frontend host. See BACKEND_DEPLOYMENT.md, SCHEDULING.md and PERSISTENCE_AUDIT.md.

Cloud independence remains DEFERRED. No service is promised to be free forever; no billing or automatic capacity purchase is configured. User approval is required before any GitHub or deployment work.

Framework reference: [official Next.js deployment documentation](https://nextjs.org/docs/app/getting-started/deploying). Deployment permissions remain separate from build readiness.
