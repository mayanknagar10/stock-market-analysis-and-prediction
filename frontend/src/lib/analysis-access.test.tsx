import { renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useAnalysisAccess } from './analysis-access';

const state = vi.hoisted(() => ({ publicAnalysis: false, operator: false }));
vi.mock('@/components/providers', () => ({ useOperator: () => ({ session: state.operator ? { authenticated: true } : null }) }));
vi.mock('./query', () => ({ useData: () => ({ data: { details: { public_analysis_enabled: state.publicAnalysis } } }) }));

describe('backend-owned analysis capability', () => {
    it('keeps local calculations protected', () => {
        state.publicAnalysis = false;
        state.operator = false;
        expect(renderHook(() => useAnalysisAccess()).result.current.enabled).toBe(false);
    });
    it('opens only backend-advertised public research calculations', () => {
        state.publicAnalysis = true;
        state.operator = false;
        expect(renderHook(() => useAnalysisAccess()).result.current).toEqual({ enabled: true, publicAnalysis: true });
    });
    it('retains an authenticated local operator path', () => {
        state.publicAnalysis = false;
        state.operator = true;
        expect(renderHook(() => useAnalysisAccess()).result.current.enabled).toBe(true);
    });
});
