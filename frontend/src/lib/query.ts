'use client';
import { useQuery } from '@tanstack/react-query';
import { useHydrated } from './hydration';
export function useData<T>(key: readonly unknown[], fn: (signal?: AbortSignal) => Promise<T>, enabled = true, staleTime = 30000) {
    const hydrated = useHydrated();
    const result = useQuery({ queryKey: key, queryFn: ({ signal }) => fn(signal), enabled: enabled && hydrated, staleTime,
        retry: (failureCount, error) => failureCount < 2 && (
            (error && typeof error === 'object' && 'code' in error && error.code === 'BACKEND_UNAVAILABLE') ||
            (error && typeof error === 'object' && 'status' in error && typeof error.status === 'number' && (error.status === 429 || error.status >= 500))
        ) });
    return hydrated ? result : { ...result, data: undefined, error: null, isPending: true, isLoading: true, isSuccess: false };
}
