import { Suspense } from 'react';
import { Markets } from '@/features/markets';
import { Skeleton } from '@/components/ui';
export default function Page() { return <Suspense fallback={<Skeleton/>}><Markets/></Suspense>; }
