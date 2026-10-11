'use client';
import { useWorkspaceMode } from '@/components/workspace-mode';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { api, type DTO } from '@/lib/api';
import { useData } from '@/lib/query';
import { number, timestamp, label } from '@/lib/format';
import { DataTable, PageTitle, QueryState, ResearchBadge, Section, Status } from '@/components/ui';
export const marketName = (source: string) => ({ market_india: 'Nifty 50', market_us: 'S&P 500', sector_india_Bank: 'Bank Nifty', sector_india_Energy: 'India Energy', sector_india_IT: 'India IT', sector_us_Technology: 'US Technology', sector_us_Financials: 'US Financials', sector_us_Energy: 'US Energy', nasdaq: 'Nasdaq', vix: 'US VIX', india_vix: 'India VIX', usd_inr: 'USD / INR', nikkei: 'Nikkei 225', hang_seng: 'Hang Seng', brent: 'Brent', wti: 'WTI', gold: 'Gold', dxy: 'US Dollar Index', us_yield: 'US 10Y yield' } as Record<string, string>)[source] || label(source);
export function MarketTable({ items, compact = false }: {
    items: DTO<'MarketItem'>[];
    compact?: boolean;
}) { return <DataTable headers={compact ? ['Reference', 'Session close', 'Change %', 'Session'] : ['Reference', 'Symbol', 'Session close', 'Change %', 'Exchange session', 'Retrieved', 'Availability']} rows={items.map(m => compact ? [marketName(m.source), number(m.price), m.change_percent == null ? 'Unavailable' : <span key="change" className={m.change_percent >= 0 ? 'positive' : 'negative'}>{number(m.change_percent)}%</span>, m.exchange_session || 'Unavailable'] : [marketName(m.source), m.symbol, number(m.price), m.change_percent == null ? 'Unavailable' : number(m.change_percent) + '%', m.exchange_session || 'Unavailable', timestamp(m.retrieved_at), <Status key="state" tone={m.status === 'AVAILABLE' ? 'neutral' : 'warning'}>{m.status}{m.error_code ? ' · ' + m.error_code : ''}</Status>])}/>; }
export function Markets() {
    const params = useSearchParams(); const reference = params.get('reference'); const q = useData(['markets'], api.markets); const { mode } = useWorkspaceMode();
    const categories = [...new Set(q.data?.items.filter(i => !i.category.toLowerCase().includes('sector')).map(i => i.category) || [])];
    return <><PageTitle title="Markets" description="Daily session monitor · India, global equities, volatility, FX & commodities"/><ResearchBadge/>{reference && <Section title={marketName(reference)} actions={<Link href="/markets">All market references</Link>}><MarketTable items={q.data?.items.filter(i => i.source === reference) || []}/></Section>}<QueryState pending={q.isPending} error={q.error} retry={() => q.refetch()}><MarketTickerStrip items={q.data?.items || []}/><div className="market-workbench"><div className="market-sectors"><Section title="Sector performance" description="Measured provider daily percentage changes"><SectorHeatmap items={q.data?.items || []}/></Section></div>{categories.map(c => <div className="market-category" key={c}><Section title={label(c)}><MarketTable compact={mode === 'simple'} items={q.data!.items.filter(m => m.category === c)}/></Section></div>)}</div></QueryState><p className="chart-note">Provider daily session labels differ across exchanges. These are completed daily references, not synchronized live quotes. Advanced view includes source symbols, retrieval times and availability.</p></>;
}

export function SectorHeatmap({ items }: { items: DTO<'MarketItem'>[] }) {
    const sectors = items.filter(i => i.category.toLowerCase().includes('sector'));
    return sectors.length ? <div className="sector-grid">{sectors.map(m => <div className="sector-cell" key={m.source}><strong>{marketName(m.source)}</strong><span className={m.change_percent == null ? 'muted' : m.change_percent < 0 ? 'negative' : 'positive'}>{m.change_percent == null ? 'Unavailable' : (m.change_percent > 0 ? '+' : '') + number(m.change_percent) + '%'}</span><small>{m.exchange_session || 'Session unavailable'} · {m.change_percent == null ? 'change unavailable' : m.status === 'AVAILABLE' ? 'daily change' : 'source unavailable'}</small></div>)}</div> : <p className="muted">Sector session changes are unavailable.</p>;
}
export function MarketTickerStrip({ items }: { items: DTO<'MarketItem'>[] }) {
    const sources = ['market_india', 'sector_india_Bank', 'market_us', 'nasdaq', 'india_vix', 'usd_inr'];
    return <div className="market-ticker-strip" aria-label="Market daily session references">{sources.map(source => { const m = items.find(i => i.source === source); return <div key={source}><span>{marketName(source)}</span><strong>{number(m?.price)}</strong><small className={m?.change_percent == null ? 'muted' : m.change_percent < 0 ? 'negative' : 'positive'}>{m?.change_percent == null ? 'Change unavailable' : (m.change_percent > 0 ? '+' : '') + number(m.change_percent) + '%'}</small></div>; })}</div>;
}
