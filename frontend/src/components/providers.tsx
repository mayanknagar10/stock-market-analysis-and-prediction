'use client';
import { createContext, useContext, useState, useEffect, useRef, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api, setOperatorBearer, type DTO } from '@/lib/api';
type OperatorContext = {
    session: DTO<'OperatorSession'> | null;
    pending: boolean;
    error: Error | null;
    connect: () => Promise<void>;
    bearer: (token: string) => Promise<void>;
    disconnect: () => Promise<void>;
};
const Context = createContext<OperatorContext | null>(null);
export function Providers({ children }: {
    children: ReactNode;
}) {
    const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } }));
    const [session, setSession] = useState<DTO<'OperatorSession'> | null>(null);
    const [pending, setPending] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const initialSessionRequest = useRef<ReturnType<typeof api.session> | null>(null);
    useEffect(() => { let mounted = true; initialSessionRequest.current ||= api.session(); void initialSessionRequest.current.then(value => { if (mounted)
        setSession(value); }).catch(() => { }); return () => { mounted = false; }; }, []);
    async function connect(token?: string) { setPending(true); setError(null); try {
        if (token !== undefined)
            setOperatorBearer(token);
        const current = token !== undefined ? await api.session() : await api.connect();
        setSession(current);
        await client.invalidateQueries({ queryKey: ['operator'] });
    }
    catch (e) {
        setOperatorBearer('');
        setSession(null);
        setError(e instanceof Error ? e : new Error('Operator connection failed.'));
    }
    finally {
        setPending(false);
    } }
    async function disconnect() { setPending(true); try {
        await api.disconnect();
    }
    catch (e) {
        setError(e instanceof Error ? e : new Error('Disconnect failed.'));
    }
    finally {
        setOperatorBearer('');
        setSession(null);
        client.removeQueries({ queryKey: ['operator'] });
        setPending(false);
    } }
    return <QueryClientProvider client={client}><Context.Provider value={{ session, pending, error, connect: () => connect(), bearer: connect, disconnect }}>{children}</Context.Provider></QueryClientProvider>;
}
export function useOperator() { const value = useContext(Context); if (!value)
    throw new Error('Operator provider missing'); return value; }
