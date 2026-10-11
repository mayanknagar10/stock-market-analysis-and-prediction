import { render, screen } from '@testing-library/react';
import { act } from 'react';
import { hydrateRoot } from 'react-dom/client';
import { renderToString } from 'react-dom/server';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Watchlist, Portfolio, Compare, Screener } from './tools';
let routeQuery = '';
const operatorState = vi.hoisted(() => ({ connected: false }));
vi.mock('next/navigation', () => ({ useSearchParams: () => new URLSearchParams(routeQuery) }));
vi.mock('@/components/providers', () => ({ useOperator: () => ({ session: operatorState.connected ? { authenticated: true } : null }) }));
vi.mock('@/lib/query', () => ({ useData: () => ({ data: { items: [], total: 100 }, isPending: false, error: null }) }));
function mount(node: React.ReactNode) { return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{node}</QueryClientProvider>); }
beforeEach(() => { localStorage.clear(); routeQuery = ''; operatorState.connected = false; });
describe('empty guest tools and protected calculations', () => { it('does not fabricate guest watchlist positions', () => { mount(<Watchlist />); expect(screen.getByText('Your watchlist is empty')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Refresh backend valuation' })).toBeDisabled(); }); it('does not assign demo portfolio holdings', () => { mount(<Portfolio />); expect(screen.getByText('No portfolio holdings')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Analyze portfolio' })).toBeDisabled(); }); it('requires explicit comparison selections and authentication', () => { mount(<Compare />); expect(screen.getByRole('button', { name: 'Run comparison' })).toBeDisabled(); expect(screen.getByText('Choose companies to compare')).toBeInTheDocument(); }); it('does not scan the universe on page open', () => { mount(<Screener />); expect(screen.getByRole('button', { name: 'Run screen' })).toBeDisabled(); expect(screen.getByText('No screen has been run')).toBeInTheDocument(); }); });

it('warns when portfolio storage reads work but writes fail', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('Quota exceeded', 'QuotaExceededError'); });
    mount(<Portfolio />);
    expect(screen.getByText('Browser storage unavailable; inputs will stay in this session.')).toBeInTheDocument();
    expect(screen.getByLabelText('Initial hypothetical capital')).toHaveValue(100000);
});

it('carries the selected company into the comparison picker', () => {
    routeQuery = 'ticker=TCS.NS';
    mount(<Compare />);
    expect(screen.getByRole('button', { name: 'Remove TCS.NS' })).toBeInTheDocument();
});

it('links saved portfolio positions to their stock risk workspace', () => {
    localStorage.setItem('stockpro.portfolio', JSON.stringify({ holdings: [{ ticker: 'TCS.NS', weight: 1 }], capital: '100000' }));
    mount(<Portfolio />);
    expect(screen.getByRole('link', { name: 'Risk for TCS.NS' })).toHaveAttribute('href', '/stocks/TCS.NS?section=risk');
});

it('hydrates an operator tool cleanly if the browser session resolves before its route', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const node = <QueryClientProvider client={client}><Screener/></QueryClientProvider>;
    const host = document.createElement('div');
    host.innerHTML = renderToString(node);
    operatorState.connected = true;
    const recoverable = vi.fn();
    let root: ReturnType<typeof hydrateRoot>;
    await act(async () => { root = hydrateRoot(host, node, { onRecoverableError: recoverable }); });
    expect(recoverable).not.toHaveBeenCalled();
    await act(async () => root!.unmount());
    client.clear();
});
