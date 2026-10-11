import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, it, expect, vi } from 'vitest';
import { Operator, JobDetail } from './operator';
import { api, type DTO } from '@/lib/api';
const { auth } = vi.hoisted(() => ({ auth: { connected: false, publicMode: false } }));
vi.mock('@/components/providers', () => ({ useOperator: () => ({ session: auth.connected ? { authenticated: true, mode: 'LOCAL_LOOPBACK', expires_at: null } : null, pending: false, error: null, connect: vi.fn(), bearer: vi.fn(), disconnect: vi.fn() }) }));
vi.mock('@/lib/query', () => ({ useData: (key: string[]) => ({ data: key[0] === 'model' ? { details: { public_analysis_enabled: auth.publicMode } } : undefined, isPending: false, error: null, refetch: vi.fn() }) }));
vi.mock('@/lib/api', () => ({ api: { jobs: vi.fn(), job: vi.fn(), enqueue: vi.fn() } }));
vi.mock('@/components/preview-connection', () => ({ PreviewConnection: () => <div>Preview backend connection</div> }));
const job: DTO<'JobRecord'> = { run_id: 'job-fixture', scheduled_at: '2026-10-09T00:00:00Z', job_type: 'verify', protocol_version: 'frozen', status: 'SUCCESS', success_count: 388, universe_size: 100, research_state: 'RESEARCH_ONLY', recommendation_allowed: false, production_promotion: 'BLOCKED' };
function mount(node: React.ReactNode) { return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{node}</QueryClientProvider>); }
beforeEach(() => { auth.connected = false; auth.publicMode = false; vi.mocked(api.jobs).mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 }); vi.mocked(api.enqueue).mockResolvedValue(job); vi.mocked(api.job).mockResolvedValue(job); });
describe('protected manual operations', () => { it('disables job submission for an unauthenticated guest', () => { mount(<Operator />); expect(screen.getByRole('button', { name: 'Submit Verify job' })).toBeDisabled(); expect(api.enqueue).not.toHaveBeenCalled(); expect(screen.getByText('Preview backend connection')).toBeInTheDocument(); expect(screen.getByText(/Cloud independence DEFERRED/)).toBeInTheDocument(); }); it('enqueues only after an explicit authenticated click', async () => { auth.connected = true; mount(<Operator />); expect(api.enqueue).not.toHaveBeenCalled(); fireEvent.click(screen.getByRole('button', { name: 'Submit Verify job' })); await waitFor(() => expect(api.enqueue).toHaveBeenCalledWith({ job_type: 'verify', origin_session: null })); }); it('retains honest final failure and null counters', () => { mount(<JobDetail job={{ ...job, status: 'FAILED', success_count: null, error_code: 'JOB_TIMEOUT' }}/>); expect(screen.getByText('FAILED')).toBeInTheDocument(); expect(screen.getByText('JOB_TIMEOUT')).toBeInTheDocument(); expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(2); }); });

it('hides operator connection and mutation controls on the public read-only API', () => {
    auth.publicMode = true;
    mount(<Operator/>);
    expect(screen.getByText('Manual research jobs are available only in the local internal console.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Connect local operator' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Submit Verify job' })).not.toBeInTheDocument();
});
