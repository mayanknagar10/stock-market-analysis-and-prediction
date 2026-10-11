import { StrictMode } from 'react';
import { render, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Providers } from './providers';
import { api } from '@/lib/api';

vi.mock('@/lib/api', () => ({ api: { session: vi.fn(async () => ({ authenticated: false })) }, setOperatorBearer: vi.fn() }));

describe('operator session bootstrap', () => {
    it('does not request the same session twice during strict-mode startup', async () => {
        vi.mocked(api.session).mockClear();
        render(<StrictMode><Providers><p>Workspace</p></Providers></StrictMode>);
        await waitFor(() => expect(api.session).toHaveBeenCalled());
        expect(api.session).toHaveBeenCalledTimes(1);
    });
});
