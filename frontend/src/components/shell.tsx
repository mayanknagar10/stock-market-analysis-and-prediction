'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { BarChart3, BookOpen, ChartNoAxesCombined, ChevronRight, ClipboardCheck, Columns3, Globe2, LayoutDashboard, Menu, Moon, PanelLeftClose, Settings2, ShieldCheck, SlidersHorizontal, Sun, Wallet, Bookmark, Newspaper } from 'lucide-react';
import { GlobalSearch } from './search';
import { Notice, Status } from './ui';
import { WorkspaceMode, ModeSwitch } from './workspace-mode';
import { api } from '@/lib/api';
import { useData } from '@/lib/query';

export function Shell({ children }: { children: ReactNode }) {
    const pathname = usePathname();
    const match = pathname.match(/^\/(stocks|forecast|risk)\/([^/]+)/);
    const [ticker, setTicker] = useState('RELIANCE.NS');
    const [theme, setTheme] = useState<'dark' | 'light'>('dark');
    const [themeRestored, setThemeRestored] = useState(false);
    const dialog = useRef<HTMLDialogElement>(null);
    const menuButton = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        try {
            const savedTheme = localStorage.getItem('stockpro.theme');
            setTheme(savedTheme === 'light' ? 'light' : 'dark');
            const savedTicker = localStorage.getItem('stockpro.ticker');
            if (savedTicker) setTicker(savedTicker);
        } catch { /* Browser storage is optional. */ }
        finally { setThemeRestored(true); }
    }, []);
    useEffect(() => {
        if (!themeRestored) return;
        document.documentElement.dataset.theme = theme;
        try { localStorage.setItem('stockpro.theme', theme); } catch { /* Keep session theme. */ }
    }, [theme, themeRestored]);

    const selected = match ? decodeURIComponent(match[2]) : ticker;
    useEffect(() => {
        const selection = pathname.match(/^\/(stocks|forecast|risk)\/([^/]+)/);
        if (!selection) return;
        const value = decodeURIComponent(selection[2]);
        setTicker(value);
        try { localStorage.setItem('stockpro.ticker', value); } catch { /* Keep session selection. */ }
    }, [pathname]);

    const health = useData(['health'], api.health);
    const groups = [
        { name: 'Markets', items: [
            { label: 'Dashboard', href: '/', icon: LayoutDashboard },
            { label: 'Markets', href: '/markets', icon: Globe2 },
        ] },
        { name: 'Research', items: [
            { label: 'Stocks', href: '/stocks/' + selected, icon: BookOpen },
            { label: 'Forecast', href: '/forecast/' + selected, icon: ChartNoAxesCombined },
            { label: 'Screener', href: '/screener', icon: SlidersHorizontal },
            { label: 'Compare', href: '/compare', icon: Columns3 },
        ] },
        { name: 'Portfolio', items: [
            { label: 'Portfolio', href: '/portfolio', icon: Wallet },
            { label: 'Watchlist', href: '/watchlist', icon: Bookmark },
        ] },
        { name: 'Tools', items: [
            { label: 'Risk', href: '/risk/' + selected, icon: ShieldCheck },
            { label: 'Factors', href: '/stocks/' + selected + '?section=factors', icon: BarChart3 },
            { label: 'News', href: '/stocks/' + selected + '?section=news', icon: Newspaper },
        ] },
        { name: 'Model', items: [
            { label: 'Validate', href: '/validate', icon: ClipboardCheck },
        ] },
    ];
    const routeLabel = pathname === '/' ? 'Market overview' : pathname.startsWith('/stocks/') ? 'Stock research' : pathname.startsWith('/forecast/') ? 'Forecast' : pathname.startsWith('/risk/') ? 'Risk' : pathname === '/validate' ? 'Validation' : pathname.slice(1).split('/')[0].replace('-', ' ');
    function close() { dialog.current?.close(); menuButton.current?.focus(); }
    const navigation = <>
        <Link className="brand" href="/" onClick={close} aria-label="StockPro dashboard"><span className="brand-mark" aria-hidden="true">SP</span><span className="brand-word">STOCKPRO<small>Market research</small></span></Link>
        <nav aria-label="Primary navigation" className="nav-groups">
            {groups.map(group => <div className="nav-group" key={group.name}><div className="nav-label">{group.name}</div>{group.items.map(item => {
                const active = pathname === item.href || (item.label === 'Stocks' && pathname.startsWith('/stocks/'));
                return <Link key={item.label} href={item.href} className={'nav-item' + (active ? ' active' : '')} aria-current={active ? 'page' : undefined} onClick={close}><item.icon size={16} strokeWidth={1.8}/><span>{item.label}</span></Link>;
            })}</div>)}
        </nav>
        <div className="nav-footer"><Link className="nav-item" href="/admin/research-status" onClick={close}><Settings2 size={16} strokeWidth={1.8}/><span>Research status</span></Link></div>
    </>;
    return <WorkspaceMode><div className="app-shell"><a href="#main-content" className="skip-link">Skip to workspace</a>
        <aside className="sidebar">{navigation}</aside>
        <dialog ref={dialog} className="nav-dialog" aria-label="Navigation" onClick={event => { if (event.target === event.currentTarget) close(); }} onCancel={close}><button className="drawer-close" onClick={close}><PanelLeftClose size={18}/>Close navigation</button>{navigation}</dialog>
        <div className="app-column"><header className="topbar"><button className="icon-button mobile-menu" aria-label="Open navigation" ref={menuButton} onClick={() => dialog.current?.showModal()}><Menu size={20}/></button><GlobalSearch/><div className="topbar-tools"><ModeSwitch/><div className="topbar-status"><Status tone="warning">Research only</Status></div><button className="icon-button theme-toggle" aria-label={theme === 'dark' ? 'Use light theme' : 'Use dark theme'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>}</button></div></header>
            <div className="contextbar"><span className="context-path"><span>STOCKPRO</span><ChevronRight size={12}/><strong>{routeLabel}</strong>{match && <><ChevronRight size={12}/><span>{selected}</span></>}</span><span className="connection-state"><span className={'health-dot ' + (health.data ? '' : 'offline')}/>{health.error ? 'API unavailable' : health.data ? 'API connected' : 'Connecting'}</span></div>
            <main id="main-content">{health.error && <div className="backend-page-notice"><Notice><strong>Backend disconnected</strong><p>Market data is unavailable while the local backend is disconnected.</p><button onClick={() => health.refetch()}>Retry connection</button></Notice></div>}{children}</main><footer className="app-footer"><span>Research model · No validated statistical edge</span><Link href="/validate">Evidence & limitations</Link></footer>
        </div>
    </div></WorkspaceMode>;
}
