'use client';
import Link from 'next/link';
import type { DTO } from '@/lib/api';
import { number, percent, timestamp } from '@/lib/format';
import { DataTable, Status } from '@/components/ui';
export function ForecastHistoryTable({ items }: {
    items: DTO<'HistoricalForecast'>[];
}) { return <DataTable caption="Saved forecasts are immutable. Realized outcomes are shown separately; sealed outcomes remain hidden." headers={['Company / horizon', 'Saved origin', 'Generated at', 'Information cutoff', 'Expected price', 'Validated P(+)', 'Research state', 'Outcome state', 'Realized return', 'Absolute error', 'Model']} rows={items.map(p => [<Link key={p.prediction_id} href={'/forecast/' + p.ticker + '?group=' + p.origin_group + '&tab=history'}>{p.ticker} · {p.horizon}s</Link>, p.origin_session, timestamp(p.generated_at), timestamp(p.information_cutoff), number(p.central_price), p.calibration_state === 'VALIDATED' && p.p_positive_calibrated != null && Number.isFinite(p.p_positive_calibrated) ? percent(p.p_positive_calibrated) : 'Unavailable', <Status key="research" tone="warning">LOW · Abstain</Status>, p.evaluation_sealed ? 'Sealed' : p.outcome_state, p.evaluation_sealed ? 'Sealed' : p.outcome?.status === 'matured' ? percent(p.outcome.actual_return) : 'Pending / unavailable', p.evaluation_sealed ? 'Sealed' : number(p.outcome?.absolute_error), p.model_version])}/>; }
