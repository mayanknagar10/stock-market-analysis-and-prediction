import Link from 'next/link';
import { EmptyState } from '@/components/ui';
export default function NotFound() { return <EmptyState title="Workspace not found" detail="Choose a workspace from the navigation or search for a company."><Link href="/">Return to dashboard</Link></EmptyState>; }
