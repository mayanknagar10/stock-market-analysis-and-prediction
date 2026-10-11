import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, it, expect, vi } from 'vitest';
import { StrictMode } from 'react';
import { Shell } from './shell';
vi.mock('next/navigation', () => ({ usePathname: () => '/stocks/TCS.NS', useRouter: () => ({ push: vi.fn() }) }));
vi.mock('./search', () => ({ GlobalSearch: () => <input aria-label="Search companies or tickers"/> }));
vi.mock('@/lib/query', () => ({ useData: (key: string[]) => ({ data: key[0] === 'health' ? { status: 'ok' } : { model_version: 'V5' } }) }));
beforeEach(() => { localStorage.clear(); delete document.documentElement.dataset.theme; HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); }; });
describe('responsive research shell', () => { it('opens and closes accessible mobile navigation', () => { render(<Shell><p>Workspace content</p></Shell>); fireEvent.click(screen.getByRole('button', { name: 'Open navigation' })); const dialog = screen.getByRole('dialog', { name: 'Navigation' }); expect(within(dialog).getByRole('navigation', { name: 'Primary navigation' })).toBeInTheDocument(); fireEvent.click(within(dialog).getByRole('button', { name: 'Close navigation' })); expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); }, 10000); it('keeps ticker context in navigation and persists theme preference', () => { render(<Shell><p>Workspace content</p></Shell>); expect(screen.getByRole('link', { name: 'Forecast' })).toHaveAttribute('href', '/forecast/TCS.NS'); fireEvent.click(screen.getByRole('button', { name: 'Use light theme' })); expect(document.documentElement).toHaveAttribute('data-theme', 'light'); expect(localStorage.getItem('stockpro.theme')).toBe('light'); }); });

describe('persisted theme startup', () => {
    it('restores a preloaded light preference in StrictMode and preserves it across reloads', () => {
        localStorage.setItem('stockpro.theme', 'light');
        const first = render(<StrictMode><Shell><p>Workspace content</p></Shell></StrictMode>);
        expect(document.documentElement).toHaveAttribute('data-theme', 'light');
        expect(localStorage.getItem('stockpro.theme')).toBe('light');
        expect(screen.getByRole('button', { name: 'Use dark theme' })).toBeInTheDocument();
        first.unmount();
        delete document.documentElement.dataset.theme;
        render(<StrictMode><Shell><p>Reloaded workspace</p></Shell></StrictMode>);
        expect(document.documentElement).toHaveAttribute('data-theme', 'light');
        expect(localStorage.getItem('stockpro.theme')).toBe('light');
        expect(screen.getByRole('button', { name: 'Use dark theme' })).toBeInTheDocument();
    });
    it('rejects unsupported saved theme values and persists the supported dark fallback', () => {
        localStorage.setItem('stockpro.theme', 'unsupported-theme');
        render(<Shell><p>Workspace content</p></Shell>);
        expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
        expect(localStorage.getItem('stockpro.theme')).toBe('dark');
        expect(screen.getByRole('button', { name: 'Use light theme' })).toBeInTheDocument();
    });
});
