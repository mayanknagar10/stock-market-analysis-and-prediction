'use client';
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Search } from 'lucide-react';
import { api, type DTO } from '@/lib/api';
import { useData } from '@/lib/query';
type MarketDestination = { name: string; market: string; href: string };
const marketDestinations: MarketDestination[] = [{ name: 'Nifty 50', market: 'India · index', href: '/markets?reference=market_india' }, { name: 'Bank Nifty', market: 'India · sector index', href: '/markets?reference=sector_india_Bank' }, { name: 'S&P 500', market: 'US · index', href: '/markets?reference=market_us' }, { name: 'Nasdaq', market: 'US · index', href: '/markets?reference=nasdaq' }, { name: 'India VIX', market: 'India · volatility', href: '/markets?reference=india_vix' }, { name: 'USD / INR', market: 'FX reference', href: '/markets?reference=usd_inr' }];
const quickRoutes = [{ name: 'Dashboard', href: '/' }, { name: 'Markets', href: '/markets' }, { name: 'Screener', href: '/screener' }, { name: 'Validate', href: '/validate' }];
export function GlobalSearch() {
    const router = useRouter();
    const [text, setText] = useState('');
    const [debounced, setDebounced] = useState('');
    const [open, setOpen] = useState(false);
    const [index, setIndex] = useState(-1);
    const [recent, setRecent] = useState<DTO<'SearchItem'>[]>([]);
    const box = useRef<HTMLDivElement>(null);
    const input = useRef<HTMLInputElement>(null);
    useEffect(() => { const keydown = (event: KeyboardEvent) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); input.current?.focus(); setOpen(true); } }; window.addEventListener('keydown', keydown); return () => window.removeEventListener('keydown', keydown); }, []);
    useEffect(() => { const id = setTimeout(() => setDebounced(text), 300); return () => clearTimeout(id); }, [text]);
    useEffect(() => { try {
        const raw: unknown = JSON.parse(localStorage.getItem('stockpro.recent') || '[]');
        if (Array.isArray(raw))
            setRecent(raw.filter((x): x is DTO<'SearchItem'> => typeof x === 'object' && x !== null && typeof x.ticker === 'string' && typeof x.company === 'string').slice(0, 6));
    }
    catch { } }, []);
    const query = useData(['search', debounced], signal => api.search(debounced, 8, signal), open && debounced.length > 0, 300000);
    const waiting = text !== debounced;
    const marketMatches = !waiting && text.trim() ? marketDestinations.filter(m => (m.name + ' ' + m.market).toLowerCase().includes(text.trim().toLowerCase())) : [];
    const items: (DTO<'SearchItem'> | MarketDestination)[] = text ? (waiting ? [] : [...query.data?.items || [], ...marketMatches]) : recent;
    function choose(item: DTO<'SearchItem'> | MarketDestination) { if ('href' in item) { setOpen(false); setText(''); setIndex(-1); router.push(item.href); return; } const next = [item, ...recent.filter(x => x.ticker !== item.ticker)].slice(0, 6); setRecent(next); try {
        localStorage.setItem('stockpro.recent', JSON.stringify(next));
        localStorage.setItem('stockpro.ticker', item.ticker);
    }
    catch { } setOpen(false); setText(''); setIndex(-1); router.push('/stocks/' + encodeURIComponent(item.ticker)); }
    return <div className="global-search" ref={box} onBlur={e => { if (!e.currentTarget.contains(e.relatedTarget))
        setOpen(false); }}><Search size={18}/><input ref={input} aria-label="Search companies or tickers" placeholder="Search ticker, company or market" role="combobox" aria-autocomplete="list" aria-expanded={open} aria-controls="ticker-results" aria-activedescendant={index >= 0 ? 'ticker-option-' + index : undefined} value={text} onFocus={() => setOpen(true)} onChange={e => { setText(e.target.value); setOpen(true); setIndex(-1); }} onKeyDown={e => { if (e.key === 'Escape') {
        setOpen(false);
        return;
    } if (e.key === 'ArrowDown') {
        e.preventDefault();
        setOpen(true);
        setIndex(i => Math.min(i + 1, items.length - 1));
    } if (e.key === 'ArrowUp') {
        e.preventDefault();
        setIndex(i => Math.max(i - 1, 0));
    } if (e.key === 'Enter' && !waiting && items[index]) {
        e.preventDefault();
        choose(items[index]);
    } }}/><kbd className="search-shortcut">Ctrl K</kbd>{open && <div className="search-popover"><div className="search-label">{text ? 'Companies & market references' : 'Recent selections'}</div><ul id="ticker-results" role="listbox">{items.map((item, i) => <li id={'ticker-option-' + i} role="option" aria-selected={index === i} key={'href' in item ? item.href : item.ticker} onMouseDown={e => e.preventDefault()} onClick={() => choose(item)}><strong>{'href' in item ? item.name : item.ticker}</strong><span>{'href' in item ? item.market : item.company + ' · ' + (item.ticker.endsWith('.NS') ? 'NSE · India' : item.ticker.endsWith('.BO') ? 'BSE · India' : 'US equity')}</span></li>)}</ul>{!items.length && <p>{text && (waiting || query.isFetching) ? 'Searching…' : query.error ? 'Search unavailable. Try again.' : text ? 'No matching companies.' : 'Search to choose your first company.'}</p>}{!text && <div className="search-routes"><div className="search-label">Go to workspace</div><div>{quickRoutes.map(route => <button key={route.href} type="button" onMouseDown={e => e.preventDefault()} onClick={() => { setOpen(false); router.push(route.href); }}>{route.name}<span>↗</span></button>)}</div></div>}<div className="search-help">↑↓ Select ticker <span>Enter Open</span><span>Esc Close</span></div></div>}</div>;
}
