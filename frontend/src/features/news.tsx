'use client';
import { api } from '@/lib/api';
import { useData } from '@/lib/query';
import { number, timestamp } from '@/lib/format';
import { DataTable, EmptyState, QueryState, Section } from '@/components/ui';
export function CompanyNews({ ticker, compact = false }: { ticker: string; compact?: boolean }) {
    const news = useData(['news', ticker], signal => api.news(ticker, signal), true, 300000);
    return <Section title="Company news & sentiment" description="Provider headlines · descriptive VADER finance-lexicon sentiment">
        <QueryState pending={news.isPending} error={news.error} retry={() => news.refetch()}>{news.data && <>
            {news.data.items.length ? <><p className="chart-note">Overall VADER score: {number(news.data.overall_score, 3)} · retrieved {timestamp(news.data.retrieved_at)}</p><div className="news-list">{news.data.items.slice(0, compact ? 3 : undefined).map((n, i) => <article className="news-item" key={n.url || i}><h3>{n.url && /^https?:\/\//.test(n.url) ? <a href={n.url} target="_blank" rel="noopener noreferrer">{n.title}</a> : n.title}</h3><div className="news-meta"><span>{n.publisher || 'Publisher unavailable'}</span><span>{timestamp(n.published_at)}</span><span>VADER {number(n.sentiment_score, 3)} · {n.sentiment_label || 'Unavailable'}</span></div></article>)}</div></> : <EmptyState title="News unavailable" detail="No eligible provider headlines or sentiment data are available."/>}
            {!compact && <p className="chart-note">Lexicon sentiment describes headline language. Advanced NLP, filings and market-wide news are not currently available.</p>}
        </>}</QueryState>
    </Section>;
}
export function CorporateEvents({ ticker }: { ticker: string }) {
    const history = useData(['events', ticker], signal => api.history(ticker, undefined, signal));
    const events = history.data?.items.filter(b => (b.dividends != null && b.dividends !== 0) || (b.stock_splits != null && b.stock_splits !== 0)) || [];
    return <Section title="Historical corporate actions" description="Provider daily history · dividends and splits, not a forward event calendar"><QueryState pending={history.isPending} error={history.error} retry={() => history.refetch()}>{events.length ? <DataTable headers={['Exchange session', 'Dividend · provider units', 'Split · provider ratio']} rows={events.map(b => [b.exchange_session, number(b.dividends), number(b.stock_splits)])}/> : <EmptyState title="No reported corporate actions" detail="No dividend or split records are available in the returned history. Scheduled earnings, filings and upcoming events require a separate backend capability."/>}</QueryState></Section>;
}
