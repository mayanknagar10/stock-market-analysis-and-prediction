import createClient from 'openapi-fetch';
import type { paths, components } from './generated/api';
export type DTO<K extends keyof components['schemas']> = components['schemas'][K];
export type Group = 'canonical' | 'legacy';
export class ApiRequestError extends Error {
    constructor(message: string, public readonly code: string, public readonly status: number, public readonly requestId?: string) {
        super(message);
        this.name = 'ApiRequestError';
    }
}
export function normalizeBaseUrl(value: string | undefined): string {
    if (!value?.trim())
        throw new ApiRequestError('API configuration missing. Set NEXT_PUBLIC_API_BASE_URL to connect a backend; market data and forecasts are unavailable in this preview.', 'BACKEND_NOT_CONFIGURED', 0);
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash)
        throw new Error('API configuration is invalid. Use an HTTP(S) backend root URL.');
    return url.toString().replace(/\/$/, '').replace(/\/api\/v1$/, '');
}
const client = createClient<paths>({ credentials: 'include', fetch: async (request) => {
        const base = normalizeBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL);
        const url = new URL(request.url);
        return fetch(base + url.pathname + url.search, {
            method: request.method,
            headers: request.headers,
            credentials: request.credentials,
            signal: request.signal,
            body: request.body ? await request.text() : undefined,
        });
    }, baseUrl: 'https://api-config.invalid' });
let operatorBearer = '';
export function setOperatorBearer(value: string) { operatorBearer = value; }
const auth = () => operatorBearer ? { Authorization: 'Bearer ' + operatorBearer } : {};
function failure(error: unknown, status: number): ApiRequestError {
    const body = error && typeof error === 'object' && 'error' in error ? error.error : null;
    const detail = body && typeof body === 'object' ? body : null;
    const message = detail && 'message' in detail && typeof detail.message === 'string'
        ? detail.message : 'The backend could not complete this request.';
    const code = detail && 'code' in detail && typeof detail.code === 'string' ? detail.code : 'API_ERROR';
    const requestId = detail && 'request_id' in detail && typeof detail.request_id === 'string' ? detail.request_id : undefined;
    return new ApiRequestError(message, code, status, requestId);
}
async function result<T>(request: Promise<{
    data?: T;
    error?: unknown;
    response: Response;
}>): Promise<T> {
    let r;
    try { r = await request; }
    catch (error) {
        if (error instanceof TypeError)
            throw new ApiRequestError('The backend is unreachable. Check the connection and retry.', 'BACKEND_UNAVAILABLE', 0);
        throw error;
    }
    if (r.error || r.data === undefined)
        throw failure(r.error, r.response.status);
    return r.data;
}
export const api = {
    health: (signal?: AbortSignal) => result(client.GET('/api/v1/health', { signal })),
    model: (signal?: AbortSignal) => result(client.GET('/api/v1/model/status', { signal })),
    dataHealth: (signal?: AbortSignal) => result(client.GET('/api/v1/data/health', { signal })),
    collection: (signal?: AbortSignal) => result(client.GET('/api/v1/collection/status', { signal })),
    system: (signal?: AbortSignal) => result(client.GET('/api/v1/system/status', { signal })),
    search: (q = '', limit = 100, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/search', { signal, params: { query: { q, limit } } })),
    markets: (signal?: AbortSignal) => result(client.GET('/api/v1/markets/context', { signal })),
    overview: (ticker: string, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/overview', { signal, params: { path: { ticker } } })),
    fundamentals: (ticker: string, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/fundamentals', { signal, params: { path: { ticker } } })),
    news: (ticker: string, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/news', { signal, params: { path: { ticker } } })),
    history: async (ticker: string, start?: string, signal?: AbortSignal) => { const read = (offset: number) => result(client.GET('/api/v1/stocks/{ticker}/history/series', { signal, params: { path: { ticker }, query: { latest: true, start, limit: 1000, offset } } })); const latest = await read(0); if (latest.total > 1000) {
        const pages = await Promise.all(Array.from({ length: Math.ceil(latest.total / 1000) - 1 }, (_, i) => read((i + 1) * 1000)));
        return { ...latest, items: [...pages.flatMap(p => p.items), ...latest.items].sort((a, b) => a.exchange_session.localeCompare(b.exchange_session)) };
    } return latest; },
    technicals: (ticker: string, group: 'trend' | 'momentum' | 'volatility' | 'volume', signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/technicals/series', { signal, params: { path: { ticker }, query: { group, limit: 252 } } })),
    risk: (ticker: string, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/risk/report', { signal, params: { path: { ticker } } })),
    factors: (ticker: string, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/factors/report', { signal, params: { path: { ticker } } })),
    forecast: (ticker: string, group: Group, signal?: AbortSignal) => result(client.GET('/api/v1/stocks/{ticker}/forecast', { signal, params: { path: { ticker }, query: { group } } })),
    forecasts: (group: Group | 'v5', ticker?: string, offset = 0, limit = 20, signal?: AbortSignal) => result(client.GET('/api/v1/research/forecasts', { signal, params: { query: { group, ticker, offset, limit } } })),
    detail: (prediction_id: string, group: Group, signal?: AbortSignal) => result(client.GET('/api/v1/research/forecasts/{prediction_id}', { signal, params: { path: { prediction_id }, query: { group } } })),
    analogues: (prediction_id: string, group: Group, signal?: AbortSignal) => result(client.GET('/api/v1/research/forecasts/{prediction_id}/analogues', { signal, params: { path: { prediction_id }, query: { group } } })),
    scenario: (prediction_id: string, body: DTO<'ScenarioRequest'>) => result(client.POST('/api/v1/research/forecasts/{prediction_id}/scenario', { params: { path: { prediction_id } }, body, headers: auth() })),
    validation: (group: Group, signal?: AbortSignal) => result(client.GET('/api/v1/validation/prospective', { signal, params: { query: { group } } })),
    historical: (report: 'final' | 'development' | 'quarantine', signal?: AbortSignal) => result(client.GET('/api/v1/validation/historical/v5/{report}', { signal, params: { path: { report } } })),
    portfolio: (body: DTO<'PortfolioRequest'>) => result(client.POST('/api/v1/analysis/portfolio', { body, headers: auth() })),
    watchlist: (body: DTO<'WatchlistRequest'>) => result(client.POST('/api/v1/analysis/watchlist', { body, headers: auth() })),
    compare: (body: DTO<'CompareRequest'>) => result(client.POST('/api/v1/analysis/compare', { body, headers: auth() })),
    screen: (body: DTO<'ScreenRequest'>) => result(client.POST('/api/v1/analysis/screen', { body, headers: auth() })),
    session: () => result(client.GET('/api/v1/operations/session', { headers: auth() })),
    connect: () => result(client.POST('/api/v1/operations/session')),
    disconnect: () => result(client.DELETE('/api/v1/operations/session', { headers: auth() })),
    operatorStatus: () => result(client.GET('/api/v1/operations/status', { headers: auth() })),
    jobs: () => result(client.GET('/api/v1/operations/jobs', { headers: auth() })),
    job: (run_id: string) => result(client.GET('/api/v1/operations/jobs/{run_id}', { params: { path: { run_id } }, headers: auth() })),
    enqueue: (body: DTO<'JobRequest'>) => result(client.POST('/api/v1/operations/jobs', { body, headers: auth() })),
};
