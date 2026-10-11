'use client';
import { ApiError } from '@/components/ui';
export default function ErrorBoundary({ error, reset }: {
    error: Error;
    reset: () => void;
}) { return <ApiError error={error} retry={reset}/>; }
