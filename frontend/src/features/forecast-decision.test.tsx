import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { DTO } from '@/lib/api';
import { ForecastDecision, ForecastEndpoints, ForecastReliability } from './forecast-decision';

const horizon = {
    prediction_id: 'saved-5', horizon: 5, predicted_log_return: .02, predicted_return: Math.expm1(.02),
    predicted_price: 102.02, quantile_log_returns: [-.1, -.05, 0, .05, .1],
    quantile_prices: [90, 95, 100, 105, 110], price_assumption: 'Total-return-equivalent implied price; future splits/dividends are not known',
    probability: .7, raw_probability: .7, calibration_state: 'UNVERIFIED', confidence: 'LOW', abstain: true,
    abstention_reasons: ['RESEARCH_ONLY'], ood_status: 'UNKNOWN', regime: 'sideways', data_quality_status: 'ELIGIBLE',
    training_cutoff: '2025-01-01T00:00:00Z', feature_schema: 'frozen', provenance_references: [], recommendation_allowed: false,
} as DTO<'HorizonForecast'>;
const forecast = {
    ticker: 'TCS.NS', current_price: 100, price_timestamp: '2026-10-07T18:30:00Z',
    prediction_origin: '2026-10-09T09:23:52Z', information_cutoff: '2026-10-09T09:23:52Z',
    price_timestamp_kind: 'DAILY_ORIGIN_SESSION_LABEL', origin_session: '2026-10-08', origin_group: 'legacy',
    model_version: 'research-20261008T190859', protocol_version: 'legacy-manual-v1', horizons: [horizon],
    research_state: 'RESEARCH_ONLY', production_promotion: 'BLOCKED', recommendation_allowed: false,
} as DTO<'ForecastResponse'>;

describe('forecast decision view', () => {
    it('prioritizes abstention and keeps unverified probability hidden', () => {
        render(<ForecastDecision forecast={forecast} selected={horizon} currency="INR" onSelect={() => {}}/>);
        expect(screen.getByText('Final overall forecast')).toBeInTheDocument();
        expect(screen.getByText('No strong statistical edge')).toBeInTheDocument();
        expect(screen.getByText('Implied total-return-equivalent price')).toBeInTheDocument();
        expect(screen.getByText('Not available', { exact: true })).toBeInTheDocument();
        expect(screen.getByText(/Origin close is the completed daily bar/)).toBeInTheDocument();
        expect(document.body.textContent).not.toMatch(/BUY|SELL|70\.00%/);
    });
    it('shows independent horizon endpoints and flags an inconsistent saved interval', () => {
        render(<ForecastEndpoints forecast={forecast} selectedHorizon={5} currency="INR"/>);
        expect(screen.getByText('Independent forecast endpoints')).toBeInTheDocument();
        expect(screen.getByText('Expected return by horizon')).toBeInTheDocument();
        const outside = { ...horizon, predicted_price: 125 };
        render(<ForecastEndpoints forecast={{ ...forecast, horizons: [outside] }} selectedHorizon={5} currency="INR"/>);
        expect(screen.getByText(/outside the saved 80% quantile band/i)).toBeInTheDocument();
    });
    it('plots matured adjusted returns on the forecast equivalent-price scale', () => {
        const history = { items: [{ prediction_id: 'saved-5', outcome: {
            status: 'matured', actual_return: .05, actual_future_price: 170,
            return_convention: 'vendor_total_return_adj_close',
        } }] } as unknown as DTO<'ResearchHistoryResponse'>;
        const { container } = render(<ForecastEndpoints forecast={forecast} selectedHorizon={5} currency="INR" history={history}/>);
        const realized = container.querySelector('.endpoint-realized');
        expect(realized).not.toBeNull();
        // The saved origin is 100; adjusted realized return of 5% is 105, not the raw close of 170.
        expect(Number(realized?.getAttribute('cx'))).toBe(118 + (105 - 90) / (110 - 90) * 490);
    });
    it('shows the real sample count and failed gate without promoting evidence', () => {
        const validation = { origin_group: 'legacy', evidence_state: 'INSUFFICIENT_FORWARD_EVIDENCE',
            groups: { horizon: { '5': { forecasts: 97, matured: 0, pending: 97, mae: null, coverage_80: null } } },
        } as unknown as DTO<'ValidationResponse'>;
        render(<ForecastReliability selected={horizon} validation={validation} originGroup="legacy"/>);
        expect(screen.getByText('Can I trust this forecast?')).toBeInTheDocument();
        expect(screen.getByText(/0 of 97 recorded forecasts matured/)).toBeInTheDocument();
        expect(screen.getByText(/V5 final gate failed/)).toBeInTheDocument();
    });
});
