'use client';
import type { ReactNode } from 'react';
import { AlertTriangle, Info, RefreshCw } from 'lucide-react';
import { isRecord, label, number } from '@/lib/format';
import { ApiRequestError } from '@/lib/api';
export function ResearchBadge() { return <div className="research-badge"><AlertTriangle size={16}/><strong>Research only</strong><span>LOW confidence · Abstain</span><span>No validated statistical edge</span></div>; }
export function Status({ children, tone = 'neutral' }: {
    children: ReactNode;
    tone?: 'neutral' | 'warning' | 'danger' | 'good';
}) { return <span className={'status ' + tone}>{children}</span>; }
export function PageTitle({ title, description, actions }: {
    title: string;
    description?: string;
    actions?: ReactNode;
}) { return <header className="page-title"><div><h1>{title}</h1>{description && <p>{description}</p>}</div>{actions && <div className="actions">{actions}</div>}</header>; }
export function Section({ title, description, children, actions }: {
    title: string;
    description?: string;
    children: ReactNode;
    actions?: ReactNode;
}) { return <section className="section"><div className="section-heading"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{actions}</div>{children}</section>; }
export function EmptyState({ title, detail, children }: {
    title: string;
    detail: string;
    children?: ReactNode;
}) { return <div className="empty-state"><Info size={22}/><h3>{title}</h3><p>{detail}</p>{children}</div>; }
const errorTitles: Record<string, string> = {
    BACKEND_UNAVAILABLE: 'Backend offline',
    BACKEND_NOT_CONFIGURED: 'Backend offline',
    INVALID_TICKER: 'Invalid ticker',
    MODEL_UNAVAILABLE: 'Forecast unavailable',
    STALE_DATA: 'Market data is stale',
    PROVIDER_FAILURE: 'Provider unavailable',
    OUTCOME_PENDING: 'Outcome pending',
};
export function ApiError({ error, retry }: {
    error: unknown;
    retry?: () => void;
}) { return <div role="alert" className="notice danger"><AlertTriangle size={18}/><div><strong>{error instanceof ApiRequestError ? errorTitles[error.code] || 'Data unavailable' : 'Data unavailable'}</strong><p>{error instanceof Error ? error.message : 'The request could not be completed.'}</p>{retry && <button onClick={retry}><RefreshCw size={14}/>Retry</button>}</div></div>; }
export function Skeleton() { return <div role="status" className="skeleton"><span>Loading data…</span><div /><div /><div /></div>; }
export function QueryState({ pending, error, retry, children }: {
    pending: boolean;
    error: unknown;
    retry?: () => void;
    children: ReactNode;
}) { if (pending)
    return <Skeleton />; if (error)
    return <ApiError error={error} retry={retry}/>; return <>{children}</>; }
export function Notice({ children, tone = 'warning' }: {
    children: ReactNode;
    tone?: 'warning' | 'danger' | 'neutral';
}) { return <div className={'notice ' + tone}><Info size={18}/><div>{children}</div></div>; }
export function Metric({ name, value, detail }: {
    name: string;
    value: ReactNode;
    detail?: ReactNode;
}) { return <div className="metric"><span>{name}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>; }
export function DataTable({ headers, rows, caption }: {
    headers: string[];
    rows: ReactNode[][];
    caption?: string;
}) { return <div className="table-scroll" tabIndex={0} role="region" aria-label={caption || 'Data table'}><table>{caption && <caption>{caption}</caption>}<thead><tr>{headers.map((h, i) => <th key={h + i} scope="col">{h}</th>)}</tr></thead><tbody>{rows.length ? rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>) : <tr><td colSpan={headers.length}>No records available.</td></tr>}</tbody></table></div>; }
export function Scalar({ value }: {
    value: unknown;
}) { if (value == null)
    return <span className="muted">Unavailable</span>; if (typeof value === 'number')
    return <>{number(value, Number.isInteger(value) ? 0 : 4)}</>; if (typeof value === 'boolean')
    return <>{value ? 'Yes' : 'No'}</>; if (typeof value === 'string')
    return <span className="wrap">{value}</span>; if (Array.isArray(value))
    return <div className="stack compact">{value.map((v, i) => <div key={i}><Scalar value={v}/></div>)}</div>; if (isRecord(value))
    return <ObjectTable value={value}/>; return <span>Unavailable</span>; }
export function ObjectTable({ value }: {
    value: unknown;
}) { if (!isRecord(value))
    return <Scalar value={value}/>; return <DataTable headers={['Measure', 'Reported value']} rows={Object.entries(value).map(([k, v]) => [label(k), <Scalar key={k} value={v}/>])}/>; }
export function Notes({ items }: {
    items: readonly string[];
}) { return <ul className="notes">{items.map((v, i) => <li key={i}>{v}</li>)}</ul>; }

export function DataHealth({ state }: { state: string | null | undefined }) { const value = state || 'Unavailable'; const healthy = ['ELIGIBLE', 'CURRENT', 'AVAILABLE', 'HEALTHY', 'OK'].includes(value.toUpperCase()); const display = value === 'SESSION_BAR_NOT_LIVE_QUOTE' ? 'Completed session · not live' : value; return <Status tone={healthy ? 'neutral' : 'warning'}>{display}</Status>; }
