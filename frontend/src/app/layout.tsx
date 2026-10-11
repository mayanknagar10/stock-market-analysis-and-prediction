import type { Metadata } from 'next';
import { Suspense } from 'react';
import { Providers } from '@/components/providers';
import { Shell } from '@/components/shell';
import './globals.css';
export const metadata: Metadata = { title: { default: 'StockPro · Research workspace', template: '%s · StockPro' }, description: 'Research-only stock analysis with immutable forecasts and explicit evidence.' };
export default function RootLayout({ children }: {
    children: React.ReactNode;
}) { return <html lang="en" suppressHydrationWarning><body><Providers><Suspense fallback={<div className="boot">Opening research workspace…</div>}><Shell>{children}</Shell></Suspense></Providers></body></html>; }
