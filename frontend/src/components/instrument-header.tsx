'use client';
import type { ReactNode } from 'react';
import { api } from '@/lib/api';
import { useData } from '@/lib/query';
import { money, timestamp } from '@/lib/format';
import { DataHealth, ResearchBadge } from './ui';
export function InstrumentHeader({ ticker, actions, compact = false }: { ticker: string; actions?: ReactNode; compact?: boolean }) {
    const company = useData(['fundamentals', ticker], signal => api.fundamentals(ticker, signal));
    const quote = useData(['overview', ticker], signal => api.overview(ticker, signal));
    const values = quote.data?.values;
    const currency = company.data?.currency || (ticker.endsWith('.NS') || ticker.endsWith('.BO') ? 'INR' : 'USD');
    const name = company.data?.company.name || ticker;
    return <header className={'instrument-header' + (compact ? ' compact-instrument' : '')}><div className="instrument-identity"><span className="instrument-symbol">{ticker}</span>{compact ? <h2>{name}</h2> : <h1>{name}</h1>}<p>{company.data?.company.exchange || 'Exchange unavailable'} · {company.data?.company.sector || 'Equity research'}</p></div><div className="instrument-quote"><strong>{money(typeof values?.current_price === 'number' ? values.current_price : null, currency)}</strong><span>{currency} · completed daily close</span><small>{typeof values?.price_timestamp === 'string' ? timestamp(values.price_timestamp) : 'Timestamp unavailable'} · not live</small></div><div className="instrument-controls"><div className="actions">{actions}</div><DataHealth state={quote.data?.data_quality_status || (quote.error ? 'UNAVAILABLE' : 'Loading')}/></div><ResearchBadge /></header>;
}
