'use client';
import { useEffect, useState } from 'react';
import { getPreviewApiBaseUrl, setPreviewApiBaseUrl } from '@/lib/api';
import { Section } from './ui';

export function PreviewConnection() {
    const [url, setUrl] = useState('');
    const [override, setOverride] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState('');
    useEffect(() => { const current = getPreviewApiBaseUrl(); if (current) { setUrl(current); setOverride(true); } }, []);

    function save() {
        try {
            setPreviewApiBaseUrl(url.trim());
            setOverride(true);
            setSaved(true);
            setError('');
        } catch (cause) { setError(cause instanceof Error ? cause.message : 'Backend URL could not be saved.'); }
    }
    function useDefault() {
        try { setPreviewApiBaseUrl(null); setUrl(''); setOverride(false); setSaved(true); setError(''); }
        catch (cause) { setError(cause instanceof Error ? cause.message : 'Backend URL could not be reset.'); }
    }

    return <Section title="Preview backend connection" description="For this browser only · development self-use">
        <p className="chart-note">When a temporary tunnel restarts, paste its new HTTPS URL here. This changes only this browser; no credential or model setting is stored. Reload to fetch data from the new endpoint.</p>
        <form className="form-panel" onSubmit={event => { event.preventDefault(); save(); }}>
            <label>Temporary HTTPS tunnel URL<input type="url" inputMode="url" autoComplete="off" value={url} onChange={event => { setUrl(event.target.value); setSaved(false); }} placeholder="https://…trycloudflare.com" required/></label>
            <div className="actions"><button className="primary" type="submit">Save backend URL</button>{override && <button type="button" onClick={useDefault}>Use deployment default</button>}</div>
        </form>
        {error && <p role="alert" className="negative">{error}</p>}
        {saved && <p className="chart-note">Connection choice saved in this browser. <a href="/admin/research-status">Reload to connect</a>.</p>}
    </Section>;
}
