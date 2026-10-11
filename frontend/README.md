# StockPro web workspace

Research-only Next.js App Router frontend for the existing local Python backend. No page trains a model or generates a new forecast. Canonical prospective, legacy/manual and reconstructed historical evidence remain distinct. V5 final failure and blocked promotion stay visible.

## Configuration

Set NEXT_PUBLIC_API_BASE_URL to the backend root, for example http://127.0.0.1:8001 during local development. A trailing /api/v1 is normalized. Production builds do not fetch data; a missing URL produces a visible API configuration error at request time. Never put a read or operator token in NEXT_PUBLIC variables. Direct reads require a backend permitting this frontend Origin; this implementation does not add a read-token proxy.

The configured backend must allow the exact frontend Origin and credentials. Local operator provisioning additionally requires its explicit development flag and loopback access. The operator bearer input is memory-only. Cookies remain HttpOnly and are never inspected by frontend code. Legacy Streamlit accounts are not migrated.

## Commands

npm run generate:api — regenerate src/lib/generated/api.d.ts from ../docs/api/openapi.json
npm run typecheck — strict TypeScript
npm test — Vitest / Testing Library behavior tests
npm run lint — ESLint
npm run build — production Next build using webpack
npm run dev — local development at port 3000
npm run test:e2e — separate root-owned Playwright checks


## Data and architecture

src/lib/api.ts is the only network layer; openapi-fetch consumes generated paths and components. Query keys include the selected ticker, evidence group, range or saved record. Shared current history caches last 30 seconds; stable saved records and metadata last five minutes. The chart pages fetch only their active datasets. Daily charts use explicit exchange-session dates and paginate backward from the latest bar to cover the requested range.

Python owns financial calculations. JavaScript formats reported units, sorts / filters rows, selects chart fields and maps values to presentation. Scenario percent inputs convert to the backend's fractional-unit contract. Driver bars scale existing contributions visually. No forecast trajectory is invented.

Lightweight Charts provides candles, volume, resize / theme observers, crosshair details and retained TradingView attribution. Recharts analytical modules are lazy loaded, with text summaries and table alternatives. Guest watchlist and portfolio inputs persist in this browser only; protected backend calculations run on explicit form submissions. A portfolio with excluded holdings is labeled as a surviving subset.

## Limits

Daily prices are session bars, not live trades. Current fundamentals and news are revised-vintage data, not verified historical point-in-time inputs. Provider and archived-model availability may leave data unavailable. Sensex has no configured backend reference and is labeled unavailable. No cloud deployment or laptop-independent collection is provided. Root-owned browser QA is reported separately.
