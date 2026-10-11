import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ResearchBadge, ApiError, EmptyState, Skeleton, ObjectTable, DataHealth } from './ui';
import { ApiRequestError } from '@/lib/api';
describe('research presentation', () => {
    it('states research-only abstention without a trading recommendation', () => { render(<ResearchBadge />); expect(screen.getByText(/No validated statistical edge/)).toBeInTheDocument(); expect(document.body.textContent).not.toMatch(/STRONG BUY|SELL|BUY/); });
    it('offers recovery for unavailable API data', () => { render(<ApiError error={new Error('Provider unavailable')} retry={() => { }}/>); expect(screen.getByText('Provider unavailable')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument(); });
    it('names offline, invalid ticker, and model-unavailable states without stack traces', () => {
        const { rerender } = render(<ApiError error={new ApiRequestError('The backend is unreachable.', 'BACKEND_UNAVAILABLE', 0)}/>);
        expect(screen.getByText('Backend offline')).toBeInTheDocument();
        rerender(<ApiError error={new ApiRequestError('Unsupported equity ticker.', 'INVALID_TICKER', 422)}/>);
        expect(screen.getByText('Invalid ticker')).toBeInTheDocument();
        rerender(<ApiError error={new ApiRequestError('No saved model result.', 'MODEL_UNAVAILABLE', 404)}/>);
        expect(screen.getByText('Forecast unavailable')).toBeInTheDocument();
    });
    it('distinguishes loading and empty results', () => { const { rerender } = render(<Skeleton />); expect(screen.getByRole('status')).toHaveTextContent('Loading'); rerender(<EmptyState title="No saved forecasts" detail="Canonical records are empty."/>); expect(screen.getByText('Canonical records are empty.')).toBeInTheDocument(); });
    it('labels null and nested report values honestly', () => { render(<ObjectTable value={{ beta: null, subset: { count: 3 } }}/>); expect(screen.getByText('Unavailable')).toBeInTheDocument(); expect(screen.getByText('3')).toBeInTheDocument(); });
    it('explains a completed session bar without exposing a raw provider code', () => { render(<DataHealth state="SESSION_BAR_NOT_LIVE_QUOTE"/>); expect(screen.getByText('Completed session · not live')).toBeInTheDocument(); expect(document.body.textContent).not.toContain('SESSION_BAR_NOT_LIVE_QUOTE'); });
});
