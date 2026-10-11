import { afterEach, describe, it, expect, vi } from 'vitest';
import { api, normalizeBaseUrl } from './api';
import { validPositions } from './guest';
describe('API configuration', () => { it('accepts roots and API prefixes consistently', () => { expect(normalizeBaseUrl('https://backend.example/api/v1/')).toBe('https://backend.example'); expect(normalizeBaseUrl('https://backend.example')).toBe('https://backend.example'); }); it('fails meaningfully without production configuration', () => { expect(() => normalizeBaseUrl(undefined)).toThrow('API configuration missing'); }); it('rejects URL credentials and unsupported protocols', () => { expect(() => normalizeBaseUrl('https://secret:token@backend.example')).toThrow(); expect(() => normalizeBaseUrl('file:///secret')).toThrow(); }); });
describe('guest watchlist validation', () => { it('accepts empty guest state without sample holdings', () => expect(validPositions([])).toBe(true)); it('accepts positive actual user inputs', () => expect(validPositions([{ ticker: 'RELIANCE.NS', quantity: 2, average_buy: 100, target: null, stop: null }])).toBe(true)); it('rejects duplicates, nonfinite and negative inputs', () => { const p = { ticker: 'AAPL', quantity: 1, average_buy: 100 }; expect(validPositions([p, p])).toBe(false); expect(validPositions([{ ...p, quantity: Infinity }])).toBe(false); expect(validPositions([{ ...p, stop: -2 }])).toBe(false); }); });

describe('structured API failures', () => {
    const previous = process.env.NEXT_PUBLIC_API_BASE_URL;
    afterEach(() => { vi.unstubAllGlobals(); process.env.NEXT_PUBLIC_API_BASE_URL = previous; });
    it('reports an unconfigured preview backend without making a network request', async () => {
        delete process.env.NEXT_PUBLIC_API_BASE_URL;
        const fetcher = vi.fn();
        vi.stubGlobal('fetch', fetcher);
        await expect(api.health()).rejects.toMatchObject({ code: 'BACKEND_NOT_CONFIGURED', status: 0 });
        expect(fetcher).not.toHaveBeenCalled();
    });
    it('retains the backend code and status for an invalid ticker', async () => {
        process.env.NEXT_PUBLIC_API_BASE_URL = 'http://127.0.0.1:8001';
        vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { code: 'INVALID_TICKER', message: 'Unsupported equity ticker.', request_id: 'request-1' } }), { status: 422, headers: { 'content-type': 'application/json' } })));
        await expect(api.overview('INVALID!')).rejects.toMatchObject({ code: 'INVALID_TICKER', status: 422, message: 'Unsupported equity ticker.' });
    });
    it('turns an unreachable backend into a recoverable named error', async () => {
        process.env.NEXT_PUBLIC_API_BASE_URL = 'http://127.0.0.1:8001';
        vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
        await expect(api.overview('TCS.NS')).rejects.toMatchObject({ code: 'BACKEND_UNAVAILABLE', status: 0 });
    });
    it('sends JSON POST bodies as bounded request data after changing the API origin', async () => {
        process.env.NEXT_PUBLIC_API_BASE_URL = 'http://127.0.0.1:8001';
        const fetcher = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({ matches: [], failures: {} }), { status: 200, headers: { 'content-type': 'application/json' } }));
        vi.stubGlobal('fetch', fetcher);
        await api.screen({ tickers: ['TCS.NS'], rsi_min: 0, rsi_max: 100, trend: 'any' });
        expect(fetcher).toHaveBeenCalledOnce();
        expect(fetcher.mock.calls[0][0]).toBe('http://127.0.0.1:8001/api/v1/analysis/screen');
        expect(fetcher.mock.calls[0][1]?.body).toBe(JSON.stringify({ tickers: ['TCS.NS'], rsi_min: 0, rsi_max: 100, trend: 'any' }));
    });
});
