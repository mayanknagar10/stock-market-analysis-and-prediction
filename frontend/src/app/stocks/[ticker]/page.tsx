import { Stocks } from '@/features/stocks';
export default async function Page({ params }: {
    params: Promise<{
        ticker: string;
    }>;
}) { const { ticker } = await params; return <Stocks ticker={decodeURIComponent(ticker).toUpperCase()}/>; }
