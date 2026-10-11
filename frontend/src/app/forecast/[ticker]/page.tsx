import { Forecast } from '@/features/forecast';
export default async function Page({ params }: {
    params: Promise<{
        ticker: string;
    }>;
}) { const { ticker } = await params; return <Forecast ticker={decodeURIComponent(ticker).toUpperCase()}/>; }
