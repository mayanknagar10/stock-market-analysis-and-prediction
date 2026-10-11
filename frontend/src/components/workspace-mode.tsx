'use client';
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
type Mode = 'simple' | 'advanced';
const ModeContext = createContext<{ mode: Mode; setMode: (mode: Mode) => void }>({ mode: 'simple', setMode: () => {} });
export function WorkspaceMode({ children }: { children: ReactNode }) {
    const [mode, setMode] = useState<Mode>('simple');
    const [restored, setRestored] = useState(false);
    useEffect(() => { try { setMode(localStorage.getItem('stockpro.mode') === 'advanced' ? 'advanced' : 'simple'); } catch {} setRestored(true); }, []);
    useEffect(() => { if (restored) { try { localStorage.setItem('stockpro.mode', mode); } catch {} } }, [mode, restored]);
    return <ModeContext.Provider value={{ mode, setMode }}>{children}</ModeContext.Provider>;
}
export function useWorkspaceMode() { return useContext(ModeContext); }
export function ModeSwitch() { const { mode, setMode } = useWorkspaceMode(); return <div className="mode-switch segmented" aria-label="Workspace detail"><button aria-pressed={mode === 'simple'} className={mode === 'simple' ? 'active' : ''} onClick={() => setMode('simple')}>Simple</button><button aria-pressed={mode === 'advanced'} className={mode === 'advanced' ? 'active' : ''} onClick={() => setMode('advanced')}>Advanced</button></div>; }
export function AdvancedDetails({ title, children }: { title: string; children: ReactNode }) { const { mode } = useWorkspaceMode(); return <details className="advanced-details" open={mode === 'advanced' || undefined} key={mode}><summary>{title}</summary>{children}</details>; }
