'use client';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { DataTable } from './ui';
import { number } from '@/lib/format';
export type ChartPoint = {
    session: string;
    [key: string]: number | string | null;
};
export default function AnalyticalChart({ points, keys, title }: {
    points: ChartPoint[];
    keys: string[];
    title: string;
}) { const colors = ['var(--accent)', 'var(--positive)', 'var(--warning)', 'var(--danger)', 'var(--muted)']; return <div><p className="chart-summary">{title} · {points.length} reported observations. Values come from the backend.</p><div style={{ width: '100%', height: 280, minWidth: 0 }} role="img" aria-label={title}><ResponsiveContainer width="100%" height="100%"><LineChart data={points}><CartesianGrid stroke="var(--grid)" vertical={false}/><XAxis dataKey="session" tick={{ fill: 'var(--muted)', fontSize: 11 }} minTickGap={35}/><YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} width={65}/><Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}/><Legend />{keys.map((k, i) => <Line key={k} type="linear" dataKey={k} stroke={colors[i % colors.length]} dot={false} strokeWidth={1.5} connectNulls={false} isAnimationActive={false}/>)}</LineChart></ResponsiveContainer></div><details className="chart-table"><summary>View chart data table</summary><DataTable headers={['Session', ...keys]} rows={points.map(p => [p.session, ...keys.map(k => typeof p[k] === 'number' ? number(p[k]) : 'Unavailable')])}/></details></div>; }
