import { Risk } from '@/features/stocks';
export default async function Page({ params }: {
    params: Promise<{
        ticker: string;
    }>;
}) { const { ticker } = await params; return <Risk ticker={decodeURIComponent(ticker).toUpperCase()}/>; }
