'use client';
import { useOperator } from '@/components/providers';
import { api } from './api';
import { useData } from './query';

/** The backend advertises stateless public analysis; local mode keeps operator protection. */
export function useAnalysisAccess() {
    const operator = useOperator();
    const status = useData(['system'], api.system, true, 60000);
    const publicAnalysis = status.data?.details?.public_analysis_enabled === true;
    return { enabled: publicAnalysis || !!operator.session?.authenticated, publicAnalysis };
}
