'use client';
import dynamic from 'next/dynamic';
import { Skeleton } from './ui';
export const AnalyticalChart = dynamic(() => import('./analytical-chart'), { ssr: false, loading: () => <Skeleton /> });
export const PriceChart = dynamic(() => import('./price-chart').then(m => m.PriceChart), { ssr: false, loading: () => <Skeleton /> });
