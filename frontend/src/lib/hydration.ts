'use client';
import { useSyncExternalStore } from 'react';
const subscribe = () => () => {};
const clientSnapshot = () => true;
const serverSnapshot = () => false;
// Each selectively hydrated boundary must first render the same pending snapshot
// as the server, even when an earlier boundary already populated shared queries.
export function useHydrated() { return useSyncExternalStore(subscribe, clientSnapshot, serverSnapshot); }
