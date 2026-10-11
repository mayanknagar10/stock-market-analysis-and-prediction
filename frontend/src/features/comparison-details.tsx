'use client';
import { useState } from 'react';
import { api } from '@/lib/api';
import { useData } from '@/lib/query';
import { number, label } from '@/lib/format';
import { DataTable, QueryState, Section, Status } from '@/components/ui';
type Category = 'fundamentals' | 'technicals' | 'risk' | 'factors' | 'forecast';
export function ComparisonDetails({ tickers }: { tickers: string[] }) {
    const [category, setCategory] = useState<Category | ''>('');
    return <Section title="Research comparison" description="Load a single research dimension for the companies you explicitly compared."><div className="toolbar"><label>Research dimension<select value={category} onChange={e => setCategory(e.target.value as Category | '')}><option value="">Choose a dimension</option>{(['fundamentals', 'technicals', 'risk', 'factors', 'forecast'] as const).map(c => <option key={c} value={c}>{label(c)}</option>)}</select></label></div>{category && <div className="comparison-grid">{tickers.map(ticker => <CompanyDimension key={ticker + category} ticker={ticker} category={category}/>)}</div>}</Section>;
}
function CompanyDimension({ ticker, category }: { ticker: string; category: Category }) {
    const fundamentals = useData(['fundamentals', ticker], signal => api.fundamentals(ticker, signal), category === 'fundamentals');
    const technicals = useData(['technicals', ticker, 'trend'], signal => api.technicals(ticker, 'trend', signal), category === 'technicals');
    const risk = useData(['risk', ticker], signal => api.risk(ticker, signal), category === 'risk');
    const factors = useData(['factors', ticker], signal => api.factors(ticker, signal), category === 'factors');
    const forecast = useData(['forecasts', 'canonical', ticker, 0], signal => api.forecasts('canonical', ticker, 0, 20, signal), category === 'forecast', 300000);
    const q = { fundamentals, technicals, risk, factors, forecast }[category];
    const values = category === 'fundamentals' ? fundamentals.data?.metrics : category === 'technicals' ? technicals.data?.items.at(-1)?.values : category === 'risk' ? risk.data?.metrics : category === 'factors' ? factors.data?.signals : undefined;
    return <div><h3>{ticker}</h3><QueryState pending={q.isPending} error={q.error} retry={() => q.refetch()}>{category === 'forecast' ? <><Status tone="warning">Research only · Abstain</Status><DataTable headers={['Saved horizon', 'Expected price', 'Outcome state']} rows={(forecast.data?.items || []).slice(0, 4).map(p => [p.horizon + ' sessions', number(p.central_price), p.outcome_state])}/><p className="chart-note">Canonical saved records only; manual evidence remains separate.</p></> : <DataTable headers={['Measure', 'Backend value']} rows={Object.entries(values || {}).filter(([key]) => !['dividend_yield_provider_value'].includes(key)).map(([key, value]) => [label(key), number(value)])}/>}</QueryState><p className="chart-note">{category === 'fundamentals' ? 'Current revised provider measurements · units follow stock Fundamentals.' : category === 'risk' ? 'Adjusted-return statistics · no client-side risk calculations.' : category === 'factors' ? 'Descriptive factor signals; regression detail in stock Factors.' : category === 'technicals' ? 'Backend trend indicators; no composite signal.' : ''}</p></div>;
}
