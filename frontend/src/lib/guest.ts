'use client';
import { useEffect, useState } from 'react';
import type { DTO } from './api';
import { isRecord } from './format';
export function validPositions(value: unknown): value is DTO<'WatchPosition'>[] { return Array.isArray(value) && value.length <= 30 && value.every(p => isRecord(p) && typeof p.ticker === 'string' && /^[A-Z0-9][A-Z0-9&.-]{0,29}$/.test(p.ticker) && typeof p.quantity === 'number' && Number.isFinite(p.quantity) && p.quantity > 0 && typeof p.average_buy === 'number' && Number.isFinite(p.average_buy) && p.average_buy > 0 && [p.target, p.stop].every(n => n == null || (typeof n === 'number' && Number.isFinite(n) && n > 0))) && new Set(value.map(p => p.ticker)).size === value.length; }
export function useGuestPositions() { const [positions, setPositions] = useState<DTO<'WatchPosition'>[]>([]); const [ready, setReady] = useState(false); const [storageError, setStorageError] = useState(''); useEffect(() => { try {
    const raw: unknown = JSON.parse(localStorage.getItem('stockpro.watchlist') || '[]');
    if (validPositions(raw))
        setPositions(raw);
    else
        setStorageError('Saved watchlist inputs are invalid. Import a valid file or add positions again.');
}
catch {
    setStorageError('Browser storage is unavailable. Inputs will last for this session only.');
} setReady(true); }, []); function save(next: DTO<'WatchPosition'>[]) { if (!validPositions(next))
    throw new Error('Enter unique tickers with positive quantities and prices.'); setPositions(next); try {
    localStorage.setItem('stockpro.watchlist', JSON.stringify(next));
    setStorageError('');
}
catch {
    setStorageError('Browser storage is unavailable. Inputs will last for this session only.');
} } return { positions, save, ready, storageError }; }
export function exportGuest(positions: DTO<'WatchPosition'>[]) { const blob = new Blob([JSON.stringify(positions, null, 2)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'stockpro-watchlist.json'; a.click(); URL.revokeObjectURL(url); }
