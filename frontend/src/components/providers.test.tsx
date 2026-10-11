import { StrictMode } from 'react';
import { render, waitFor } from '@testing-library/react';
import { beforeEach, describe, it, expect, vi } from 'vitest';
import { Providers } from './providers';
import { api } from '@/lib/api';

const state = vi.hoisted(() => ({ publicAnalysis: false }));
vi.mock('@/lib/api', () => ({ api: {
    model: vi.fn(async () => ({ details: { public_analysis_enabled: state.publicAnalysis } })),
    session: vi.fn(async () => ({ authenticated: false })),
}, setOperatorBearer: vi.fn() }));

describe('operator session bootstrap', () => {
    beforeEach(() => { state.publicAnalysis = false; vi.mocked(api.model).mockClear(); vi.mocked(api.session).mockClear(); });
    it('does not request the same session twice during strict-mode startup', async () => {
        render(<StrictMode><Providers><p>Workspace</p></Providers></StrictMode>);
        await waitFor(() => expect(api.session).toHaveBeenCalled());
        expect(api.session).toHaveBeenCalledTimes(1);
    });
    it('does not request an operator session from a public read-only backend', async () => {
        state.publicAnalysis = true;
        render(<Providers><p>Workspace</p></Providers>);
        await waitFor(() => expect(api.model).toHaveBeenCalled());
        await new Promise(resolve => setTimeout(resolve, 50));
        expect(api.session).not.toHaveBeenCalled();
    });
});
