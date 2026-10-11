import { act } from 'react';
import { hydrateRoot, type Root } from 'react-dom/client';
import { renderToString } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { useData } from './query';

describe('shared query hydration', () => {
    it('hydrates the server loading snapshot even if another boundary populated the cache first', async () => {
        const query = vi.fn(async () => 'fresh result');
        function Consumer() { const state = useData(['shared-startup'], query); return <p>{state.isPending ? 'Loading data…' : state.data}</p>; }
        const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
        const tree = <QueryClientProvider client={client}><Consumer/></QueryClientProvider>;
        const container = document.createElement('div');
        container.innerHTML = renderToString(tree);
        expect(container.textContent).toBe('Loading data…');
        expect(query).not.toHaveBeenCalled();
        client.setQueryData(['shared-startup'], 'cached before selective hydration');
        const recoverable = vi.fn();
        let root: Root;
        await act(async () => { root = hydrateRoot(container, tree, { onRecoverableError: recoverable }); });
        expect(recoverable).not.toHaveBeenCalled();
        expect(container.textContent).toBe('cached before selective hydration');
        expect(query).not.toHaveBeenCalled();
        await act(async () => root!.unmount());
        client.clear();
    });
    it('does not repeat a permanent invalid-ticker request', async () => {
        let requests = 0;
        function Consumer() {
            const state = useData(['invalid-ticker'], async () => {
                requests++;
                throw Object.assign(new Error('Unsupported equity ticker.'), { code: 'INVALID_TICKER', status: 422 });
            });
            return <p>{state.error ? 'Ticker rejected' : 'Loading'}</p>;
        }
        const client = new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } });
        render(<QueryClientProvider client={client}><Consumer/></QueryClientProvider>);
        await waitFor(() => expect(screen.getByText('Ticker rejected')).toBeInTheDocument());
        expect(requests).toBe(1);
        client.clear();
    });
});
