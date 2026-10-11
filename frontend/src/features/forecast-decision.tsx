'use client';
import type { DTO, Group } from '@/lib/api';
import { label, money, number, percent, timestamp } from '@/lib/format';
import { Status } from '@/components/ui';

type Forecast = DTO<'ForecastResponse'>;
type Horizon = DTO<'HorizonForecast'>;
type History = DTO<'ResearchHistoryResponse'>;
type Detail = DTO<'ArchivedDetail'>;
type Validation = DTO<'ValidationResponse'>;
type Markets = DTO<'MarketsResponse'>;

function verifiedProbability(horizon: Horizon) {
    return horizon.calibration_state === 'VALIDATED' && horizon.probability != null && Number.isFinite(horizon.probability)
        ? percent(horizon.probability) : 'Not available';
}
function band(horizon: Horizon) {
    const q = horizon.quantile_prices;
    return q.length === 5 && q.every(Number.isFinite) && q.every((v, i) => i === 0 || v >= q[i - 1]) ? q : null;
}
function metric(row: Record<string, unknown> | undefined, key: string) {
    const value = row?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function ForecastDecision({ forecast, selected, currency, onSelect }: {
    forecast: Forecast; selected: Horizon; currency: string; onSelect: (horizon: number) => void;
}) {
    const q = band(selected);
    const noEdge = selected.abstain || selected.confidence === 'LOW';
    const equivalent = selected.price_assumption?.toLowerCase().includes('total-return');
    return <section className="final-forecast" aria-label="Final overall forecast">
        <div className="final-forecast-head"><div><span className="forecast-kicker">Saved quantitative model · research only</span><h2>Final overall forecast</h2></div>
            <div className="forecast-horizon-picker" aria-label="Select forecast horizon">{forecast.horizons.map(h => <button key={h.prediction_id} aria-pressed={h.horizon === selected.horizon} className={h.horizon === selected.horizon ? 'active' : ''} onClick={() => onSelect(h.horizon)}>{h.horizon}D</button>)}</div></div>
        <div className="final-forecast-state"><strong>{noEdge ? 'No strong statistical edge' : selected.predicted_return > 0 ? 'Positive research estimate' : selected.predicted_return < 0 ? 'Negative research estimate' : 'Neutral research estimate'}</strong><Status tone="warning">{selected.confidence} confidence · {selected.abstain ? 'Abstain' : 'Research only'}</Status></div>
        <div className="final-forecast-metrics"><div><span>Origin close</span><strong>{money(forecast.current_price, currency)}</strong></div><div><span>{selected.horizon} trading-session expected return</span><strong>{percent(selected.predicted_return)}</strong></div><div><span>{equivalent ? 'Implied total-return-equivalent price' : 'Implied price'}</span><strong>{money(selected.predicted_price, currency)}</strong></div><div><span>80% saved quantile band</span><strong>{q ? money(q[0], currency) + ' – ' + money(q[4], currency) : 'Unavailable'}</strong></div><div><span>Validated P(up)</span><strong>{verifiedProbability(selected)}</strong></div></div>
        <p className="forecast-meaning">{equivalent ? 'The implied price applies an adjusted-return forecast to the preserved raw origin close; future dividends and splits are unknown. Current provider charts may reflect later revisions. It is not a guaranteed future trading price.' : 'This is a saved model estimate, not a future market quote.'}{!q && ' Saved quantiles failed ordering checks; the range is withheld.'}</p>
        <div className="final-forecast-times"><span>Forecast issued {timestamp(forecast.prediction_origin)}</span><span>Information cutoff {timestamp(forecast.information_cutoff)}</span><span>Origin session {forecast.origin_session} · Asia/Kolkata</span><span>{forecast.price_timestamp_kind === 'DAILY_ORIGIN_SESSION_LABEL' ? 'Origin close is the completed daily bar; no intraday quote time' : 'Origin price timestamp ' + timestamp(forecast.price_timestamp)}</span></div>
    </section>;
}

export function ForecastEndpoints({ forecast, selectedHorizon, currency, history }: {
    forecast: Forecast; selectedHorizon: number; currency: string; history?: History;
}) {
    const horizons = forecast.horizons.slice().sort((a, b) => a.horizon - b.horizon);
    const realizedEquivalent = (h: Horizon) => {
        const outcome = history?.items.find(item => item.prediction_id === h.prediction_id)?.outcome;
        return outcome?.status === 'matured' && outcome.return_convention === 'vendor_total_return_adj_close' && typeof outcome.actual_return === 'number' && Number.isFinite(outcome.actual_return)
            ? forecast.current_price * (1 + outcome.actual_return) : null;
    };
    const actualPrices = horizons.map(realizedEquivalent).filter((value): value is number => value !== null && Number.isFinite(value));
    const prices = [forecast.current_price, ...horizons.flatMap(h => [h.predicted_price, ...(band(h) || [])]), ...actualPrices].filter(Number.isFinite);
    const low = Math.min(...prices), high = Math.max(...prices);
    const span = Math.max(high - low, Math.abs(forecast.current_price) * .01, 1);
    const x = (value: number) => 118 + (value - low) / span * 490;
    const maxReturn = Math.max(.005, ...horizons.map(h => Math.abs(h.predicted_return)));
    return <section className="endpoint-view" aria-label="Independent forecast endpoints">
        <div className="section-heading"><div><h2>Independent forecast endpoints</h2><p>Saved 80% quantile bands · total-return-equivalent price scale</p></div></div>
        <svg className="endpoint-chart" viewBox="0 0 760 260" role="img" aria-label="Origin close and independent one, five, ten and twenty trading-session forecast ranges; no daily path is predicted">
            <line x1={x(forecast.current_price)} x2={x(forecast.current_price)} y1="26" y2="226" className="endpoint-origin-line"/>
            <text x={x(forecast.current_price)} y="17" className="endpoint-origin-label" textAnchor="middle">Origin {money(forecast.current_price, currency)}</text>
            {horizons.map((h, i) => { const q = band(h); const y = 57 + i * 45; const actual = realizedEquivalent(h);
                return <g key={h.prediction_id} className={h.horizon === selectedHorizon ? 'endpoint-selected' : ''}><text x="8" y={y + 4} className="endpoint-row-label">{h.horizon}D</text><line x1="118" x2="608" y1={y} y2={y} className="endpoint-row-guide"/>{q && <><line x1={x(q[0])} x2={x(q[4])} y1={y} y2={y} className="endpoint-range"/><line x1={x(q[1])} x2={x(q[3])} y1={y} y2={y} className="endpoint-inner"/><circle cx={x(q[0])} cy={y} r="3" className="endpoint-cap"/><circle cx={x(q[4])} cy={y} r="3" className="endpoint-cap"/></>}<circle cx={x(h.predicted_price)} cy={y} r="5" className="endpoint-estimate"/>{actual !== null && actual >= low && actual <= high && <circle cx={x(actual)} cy={y} r="4" className="endpoint-realized"/>}<text x="748" y={y + 4} textAnchor="end" className="endpoint-value">{money(h.predicted_price, currency)}</text></g>; })}
            <text x="118" y="249" className="endpoint-axis">{money(low, currency)}</text><text x="608" y="249" textAnchor="end" className="endpoint-axis">{money(high, currency)}</text>
        </svg>
        <p className="chart-note">Dots are separate horizon estimates. Whiskers are saved 10th–90th percentiles; thick segments are 25th–75th percentiles. The vertical line is the raw origin close. Hollow dots, when present, show later realized total returns on the same equivalent scale. No intervening daily path is predicted.</p>
        {horizons.some(h => { const q = band(h); return q && (h.predicted_price < q[0] || h.predicted_price > q[4]); }) && <p className="chart-note warning-text">A central estimate lies outside the saved 80% quantile band. The original saved values are shown without adjustment.</p>}
        <div className="return-comparison"><h3>Expected return by horizon</h3>{horizons.map(h => <div className="return-row" key={h.prediction_id}><span>{h.horizon}D</span><div className="return-track"><i className="return-zero"/><i className="return-dot" style={{ left: `${50 + h.predicted_return / maxReturn * 43}%` }}/></div><strong>{percent(h.predicted_return)}</strong></div>)}</div>
    </section>;
}

export function ForecastReliability({ selected, validation, originGroup }: {
    selected: Horizon; validation?: Validation; originGroup: Group;
}) {
    const row = validation?.groups.horizon?.[String(selected.horizon)] as Record<string, unknown> | undefined;
    const recorded = metric(row, 'forecasts'), matured = metric(row, 'matured');
    const mae = metric(row, 'mae'), coverage = metric(row, 'coverage_80');
    return <section className="forecast-evidence" aria-label="Forecast reliability"><h2>Can I trust this forecast?</h2><p className="forecast-evidence-answer">No validated statistical edge. The V5 final gate failed and production remains blocked.</p>
        <div className="trust-facts"><div><span>Horizon evidence</span><strong>{matured == null || recorded == null ? 'Loading saved evidence' : `${matured} of ${recorded} recorded forecasts matured`}</strong></div><div><span>Mean absolute error</span><strong>{mae == null ? 'Not available' : percent(mae)}</strong></div><div><span>80% band coverage</span><strong>{coverage == null ? 'Not available' : percent(coverage)}</strong></div><div><span>Probability calibration</span><strong>{verifiedProbability(selected) === 'Not available' ? 'Unverified · withheld' : 'Validated for this record'}</strong></div></div>
        <p className="chart-note">{originGroup === 'legacy' ? 'Preserved manual prospective stream; the counts span its recorded stocks, not this ticker alone.' : 'Canonical prospective stream; counts include only recorded forecasts.'} {matured === 0 ? 'No matured outcomes exist for this horizon.' : 'Repeated stocks and overlapping horizons reduce independent support.'} Confidence is {selected.confidence}; this forecast {selected.abstain ? 'abstains' : 'is research only'}. Historical training availability is not certified point-in-time.</p>
    </section>;
}

export function ForecastDriversContext({ detail, markets, selected }: {
    detail?: Detail; markets?: Markets; selected: Horizon;
}) {
    const entries = detail?.drivers?.features.map((feature, i) => ({ feature, value: detail.drivers!.values[i] })).filter(x => Number.isFinite(x.value)) || [];
    const positive = entries.filter(x => x.value > 0).sort((a, b) => b.value - a.value).slice(0, 3);
    const negative = entries.filter(x => x.value < 0).sort((a, b) => a.value - b.value).slice(0, 3);
    const references = (markets?.items || []).filter(x => ['market_india', 'india_vix', 'usd_inr', 'brent'].includes(x.source) && x.status === 'AVAILABLE');
    return <div className="forecast-context-grid"><section className="forecast-driver-preview"><h2>Why does the model lean this way?</h2><p className="chart-note">Saved feature contributions for {selected.horizon}D, in model log-return units. Associations are not causal explanations.</p><div className="driver-duo"><div><h3>Supports estimate</h3>{positive.length ? positive.map(x => <p key={x.feature}><span>{label(x.feature)}</span><strong className="positive">+{number(x.value, 4)}</strong></p>) : <p className="muted">Saved supporting drivers unavailable.</p>}</div><div><h3>Opposes estimate</h3>{negative.length ? negative.map(x => <p key={x.feature}><span>{label(x.feature)}</span><strong className="negative">{number(x.value, 4)}</strong></p>) : <p className="muted">Saved opposing drivers unavailable.</p>}</div></div></section>
        <section className="forecast-current-context"><h2>Current context</h2><p className="chart-note">Latest available provider observations; dates and freshness vary. These do not change the saved numeric forecast.</p>{references.length ? <div className="context-rows">{references.map(item => <div key={item.source}><span>{label(item.source.replace('market_india', 'Nifty 50').replace('india_vix', 'India VIX').replace('usd_inr', 'USD / INR'))}</span><strong>{item.change_percent == null ? 'Change unavailable' : `${item.change_percent >= 0 ? '+' : ''}${number(item.change_percent)}%`}</strong><small>{item.exchange_session || 'Session unavailable'}</small></div>)}</div> : <p className="muted">Market context is loading or unavailable.</p>}<p className="context-alignment">Context alignment: not assessed. No validated contextual adjustment is applied.</p></section>
        <section className="forecast-invalidation"><h2>What could change this view?</h2><p>Changes after the information cutoff belong to a new, separately recorded forecast. A shift in the model inputs{entries.length ? `, including ${entries.slice().sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 2).map(x => label(x.feature)).join(' and ')},` : ''} or in market conditions could change a future estimate. No validated numeric invalidation threshold is available.</p></section>
    </div>;
}
