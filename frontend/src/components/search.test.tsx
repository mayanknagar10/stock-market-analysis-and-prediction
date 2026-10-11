import { act, fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest';
import { GlobalSearch } from './search';
import { api } from '@/lib/api';
const { push } = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock('next/navigation', () => ({ useRouter: () => ({ push }) }));
vi.mock('@/lib/api', () => ({ api: { search: vi.fn() } }));
const company = { ticker: 'RELIANCE.NS', company: 'Reliance Industries', industry: 'Energy', isin: 'IN001' };
beforeEach(() => { localStorage.clear(); vi.useFakeTimers(); vi.mocked(api.search).mockResolvedValue({ items: [company], total: 1, limit: 8, offset: 0 }); });
afterEach(() => vi.useRealTimers());
function mount() { return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><GlobalSearch /></QueryClientProvider>); }
describe('keyboard company search', () => { it('debounces requests by 300ms and supports arrow/enter selection', async () => { mount(); const input = screen.getByRole('combobox'); fireEvent.focus(input); fireEvent.change(input, { target: { value: 'Reli' } }); await act(async () => { await vi.advanceTimersByTimeAsync(299); }); expect(api.search).not.toHaveBeenCalled(); await act(async () => { await vi.advanceTimersByTimeAsync(301); }); expect(api.search).toHaveBeenCalledWith('Reli', 8, expect.any(AbortSignal)); await act(async () => { await vi.advanceTimersByTimeAsync(10); }); expect(screen.getByRole('option')).toHaveTextContent('Reliance Industries'); fireEvent.keyDown(input, { key: 'ArrowDown' }); fireEvent.keyDown(input, { key: 'Enter' }); expect(push).toHaveBeenCalledWith('/stocks/RELIANCE.NS'); expect(localStorage.getItem('stockpro.recent')).toContain('RELIANCE.NS'); }); it('supports recent selection and Escape dismissal', () => { localStorage.setItem('stockpro.recent', JSON.stringify([company])); mount(); const input = screen.getByRole('combobox'); fireEvent.focus(input); expect(screen.getByRole('option')).toHaveTextContent('Reliance Industries'); fireEvent.keyDown(input, { key: 'Escape' }); expect(input).toHaveAttribute('aria-expanded', 'false'); }); });

describe('search query consistency', () => { it('removes old matches while the next query is debouncing', async () => { push.mockClear(); mount(); const input = screen.getByRole('combobox'); fireEvent.focus(input); fireEvent.change(input, { target: { value: 'Reli' } }); await act(async () => { await vi.advanceTimersByTimeAsync(300); }); await act(async () => { await vi.advanceTimersByTimeAsync(20); }); expect(screen.getByRole('option')).toHaveTextContent('NSE · India'); fireEvent.keyDown(input, { key: 'ArrowDown' }); fireEvent.change(input, { target: { value: 'TCS' } }); expect(screen.queryByRole('option')).not.toBeInTheDocument(); expect(screen.getByText('Searching…')).toBeInTheDocument(); fireEvent.keyDown(input, { key: 'Enter' }); expect(push).not.toHaveBeenCalled(); }); });

describe('market discovery', () => { it('routes a named market search to its addressable reference', async () => { push.mockClear(); vi.mocked(api.search).mockResolvedValue({ items: [], total: 0, limit: 8, offset: 0 }); mount(); const input = screen.getByRole('combobox'); fireEvent.focus(input); fireEvent.change(input, { target: { value: 'Nasdaq' } }); await act(async () => { await vi.advanceTimersByTimeAsync(300); }); await act(async () => { await vi.advanceTimersByTimeAsync(20); }); expect(screen.getByRole('option')).toHaveTextContent('US · index'); fireEvent.keyDown(input, { key: 'ArrowDown' }); fireEvent.keyDown(input, { key: 'Enter' }); expect(push).toHaveBeenCalledWith('/markets?reference=nasdaq'); }); });

describe('command access', () => {
    it('opens search with Ctrl+K and closes with Escape', () => {
        mount();
        const input = screen.getByRole('combobox');
        fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
        expect(input).toHaveFocus();
        expect(input).toHaveAttribute('aria-expanded', 'true');
        fireEvent.keyDown(input, { key: 'Escape' });
        expect(input).toHaveAttribute('aria-expanded', 'false');
    });
});
