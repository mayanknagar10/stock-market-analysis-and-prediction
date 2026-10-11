'use client';
import Link from 'next/link';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowUpRight, BarChart3, CircleAlert, Clock3 } from 'lucide-react';
import { api, type DTO } from '@/lib/api';
import { useData } from '@/lib/query';
import { useHydrated } from '@/lib/hydration';
import { useGuestPositions } from '@/lib/guest';
import { isRecord, money, number } from '@/lib/format';
import { EmptyState, Metric, Notice, QueryState, Section, Scalar, DataTable } from '@/components/ui';
import { AdvancedDetails } from '@/components/workspace-mode';
import { MarketTable, MarketTickerStrip, SectorHeatmap, marketName } from './markets';
import { ForecastHistoryTable } from './history';

export function dataHealthColor(health: string | undefined) {
    const state = health?.toUpperCase();
    if (state === 'HEALTHY' || state === 'OK') return 'var(--positive)';
    if (!state || ['UNKNOWN', 'UNAVAILABLE', 'NOT_STARTED'].includes(state)) return 'var(--muted)';
    return 'var(--warning)';
}

function collectionLabel(health: string | undefined) {
    if (health === 'HEALTHY') return 'Collection healthy';
    if (health === 'DEGRADED') return 'Collection needs attention';
    if (health === 'FAILED') return 'Collection failed';
    if (health === 'NOT_STARTED') return 'No completed batch yet';
    return 'Collection status unavailable';
}

function changeLabel(value: number | null | undefined) {
    if (value == null) return 'Change unavailable';
    return (value > 0 ? '+' : '') + number(value) + '%';
}

function ReferenceMoves({ items }: { items: DTO<'MarketItem'>[] }) {
    const sources = ['market_india', 'sector_india_Bank', 'market_us', 'nasdaq', 'india_vix', 'usd_inr'];
    const measured = sources.map(source => items.find(item => item.source === source)).filter((item): item is DTO<'MarketItem'> => Boolean(item && item.change_percent != null && item.status === 'AVAILABLE'));
    const scale = Math.max(.01, ...measured.map(item => Math.abs(item.change_percent || 0)));
    return <div className="reference-moves"><div className="deck-subhead"><span>Reference moves</span><small>Measured daily change · relative bar scale</small></div>{measured.length ? <div className="reference-move-list">{measured.map(item => {
        const change = item.change_percent || 0;
        return <div className="reference-move" key={item.source}><span>{marketName(item.source)}</span><div className="reference-move-track" aria-hidden="true"><i className={change < 0 ? 'negative-bar' : 'positive-bar'} style={{ width: Math.max(2, Math.abs(change) / scale * 100) + '%' }}/></div><strong className={change < 0 ? 'negative' : 'positive'}>{changeLabel(change)}</strong></div>;
    })}</div> : <div className="reference-move-empty">Daily changes are unavailable for these references.</div>}</div>;
}

export function Dashboard() {
    const markets = useData(['markets'], api.markets);
    const forecasts = useData(['forecasts', 'canonical', undefined, 0], signal => api.forecasts('canonical', undefined, 0, 20, signal), true, 300000);
    const canonical = useData(['validation', 'canonical'], signal => api.validation('canonical', signal), true, 300000);
    const legacy = useData(['validation', 'legacy'], signal => api.validation('legacy', signal), true, 300000);
    const health = useData(['data-health'], api.dataHealth);
    const collection = useData(['collection'], api.collection);
    const model = useData(['model'], api.model, true, 300000);
    const universe = useData(['universe'], signal => api.search('', 100, signal), true, 300000);
    const { positions } = useGuestPositions();
    const client = useQueryClient();
    const hydrated = useHydrated();
    const portfolio = hydrated ? client.getQueryData<DTO<'PortfolioResponse'>>(['portfolio-snapshot']) : undefined;
    const tracked = hydrated ? client.getQueryData<DTO<'WatchlistResponse'>>(['watchlist', positions]) : undefined;
    const items = markets.data?.items || [];
    const nifty = items.find(item => item.source === 'market_india');
    const count = (data: DTO<'ValidationResponse'> | undefined, key: string) => isRecord(data?.counts) ? <Scalar value={data.counts[key]}/> : 'Unavailable';

    return <div className="dashboard-terminal premium-dashboard">
        <div className="dashboard-heading"><div><h1>Dashboard</h1><p>India and global daily references. Company research is one step away.</p></div><Link className="quiet-link" href="/markets">Explore all markets <ArrowUpRight size={14}/></Link></div>
        <QueryState pending={markets.isPending} error={markets.error} retry={() => markets.refetch()}>{markets.data && <MarketTickerStrip items={items}/>}</QueryState>
        <div className="dashboard-lead">
            <section className="pulse-deck" aria-label="Market pulse"><div className="deck-heading"><div><h2>Market pulse</h2></div><span className="deck-asof"><Clock3 size={13}/> Latest completed sessions</span></div><div className="pulse-content"><div className="pulse-primary"><span className="pulse-region">INDIA · NIFTY 50</span><strong>{nifty?.price == null ? '—' : number(nifty.price)}</strong><span className={'pulse-change ' + (nifty?.change_percent == null ? 'muted' : nifty.change_percent < 0 ? 'negative' : 'positive')}>{changeLabel(nifty?.change_percent)}</span><p>{nifty?.exchange_session ? 'Exchange session ' + nifty.exchange_session : 'Session date unavailable'} · {nifty?.status === 'AVAILABLE' ? 'provider daily bar' : 'availability unverified'}</p></div><ReferenceMoves items={items}/></div><div className="pulse-footnote"><CircleAlert size={13}/> These are completed daily reference observations. Exchanges have different cutoffs; prices are not synchronized live quotes.</div></section>
            <section className="sector-deck" aria-label="Sector lens"><div className="deck-heading"><div><h2>Sector lens</h2></div><Link href="/markets">Open markets <ArrowUpRight size={13}/></Link></div><p className="deck-description">Available sector index changes, measured over each provider session.</p><SectorHeatmap items={items}/></section>
        </div>
        <div className="dashboard-context"><section className="global-cues"><div className="deck-heading"><div><h2>Global cues</h2></div><span className="deck-asof">Latest available daily references</span></div><MarketTable compact items={items.filter(item => ['market_us', 'nasdaq', 'nikkei', 'hang_seng', 'india_vix', 'usd_inr', 'brent', 'gold'].includes(item.source))}/></section><section className="trust-deck"><div className="deck-heading"><div><h2>Evidence, not advice</h2></div><Link href="/validate">Validate <ArrowUpRight size={13}/></Link></div><p className="trust-statement">No validated statistical edge</p><p className="trust-copy">The frozen V5 final gate failed. Saved forecasts remain research only and can abstain.</p><div className="trust-register"><div><span>Model state</span><strong>{model.data?.research_state === 'RESEARCH_ONLY' ? 'Research only' : model.data?.research_state || 'Unavailable'}</strong></div><div><span>Production use</span><strong>{model.data?.production_promotion === 'BLOCKED' ? 'Blocked' : model.data?.production_promotion || 'Unavailable'}</strong></div><div><span>New canonical collection</span><strong>{count(canonical.data, 'forecasts')} records</strong></div></div><div className="trust-metrics"><div><span>Preserved forecasts</span><strong>{count(legacy.data, 'forecasts')}</strong></div><div><span>Matured outcomes</span><strong>{count(legacy.data, 'matured')}</strong></div><div><span>Frozen universe</span><strong>{universe.data?.total ?? '—'}</strong></div></div><div className="trust-footer"><span className="health-dot" aria-hidden="true" style={{ background: dataHealthColor(health.data?.health) }}/><span>{collectionLabel(collection.data?.health)}</span></div>{(canonical.error || legacy.error) && <Notice>Evidence counts are unavailable. Open Validate to retry.</Notice>}</section></div>
        <div className="dashboard-lower"><Section title="Watchlist" description="Your tracked securities · stored in this browser" actions={<Link href="/watchlist">Manage <ArrowUpRight size={13}/></Link>}>{positions.length ? tracked ? <DataTable headers={['Ticker', 'Session close', 'Daily %', 'Status']} rows={tracked.items.slice(0, 6).map(position => [<Link key={position.ticker} href={'/stocks/' + position.ticker}>{position.ticker}</Link>, money(position.last_price, position.currency), position.change_percent == null ? 'Unavailable' : number(position.change_percent) + '%', position.status])}/> : <p className="chart-note">{positions.map(position => position.ticker).join(' · ')} · Refresh valuation in Watchlist.</p> : <EmptyState title="No watchlist positions yet" detail="Add companies to track their completed session context."><Link href="/watchlist">Open Watchlist →</Link></EmptyState>}</Section>
            <Section title="Recent forecasts" description="Recorded before outcome · canonical stream" actions={<Link href="/validate">Evidence <ArrowUpRight size={13}/></Link>}><QueryState pending={forecasts.isPending} error={forecasts.error} retry={() => forecasts.refetch()}>{forecasts.data?.items.length ? <ForecastHistoryTable items={forecasts.data.items.slice(0, 6)}/> : <EmptyState title="No forward forecasts recorded yet" detail="Saved legacy/manual forecasts remain separate from the canonical prospective study."><Link href="/forecast/RELIANCE.NS?group=legacy">View preserved research →</Link></EmptyState>}</QueryState></Section>
        </div>
        <div className="dashboard-end"><section><BarChart3 size={17}/><div><strong>Continue with a company</strong><span>Charts, forecasts, technicals and fundamentals in one stock workspace.</span></div><Link href="/stocks/RELIANCE.NS">Open stock research →</Link></section><section><div><strong>Portfolio snapshot</strong><span>Last explicitly calculated hypothetical allocation.</span></div>{portfolio ? <span className="portfolio-inline">{money(portfolio.series.at(-1)?.value, portfolio.currency)}</span> : <Link href="/portfolio">Create analysis →</Link>}</section></div>
        <AdvancedDetails title="Collection, model & data coverage"><div className="metric-strip"><Metric name="Research universe" value={universe.data?.total ?? 'Unavailable'}/><Metric name="Collection" value={collectionLabel(collection.data?.health)}/><Metric name="Completed batches" value={collection.data?.batches.length ?? 'Unavailable'}/><Metric name="Preserved manual forecasts" value={count(legacy.data, 'forecasts')}/></div><p className="chart-note">Local backend required. Cloud collection is deferred.</p><Link href="/admin/research-status">Research status →</Link></AdvancedDetails>
    </div>;
}
