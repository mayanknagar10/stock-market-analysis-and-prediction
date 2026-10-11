export const number = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? 'Unavailable' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(v);
export const percent = (v: number | null | undefined) => v == null ? 'Unavailable' : number(v * 100) + '%';
export const money = (v: number | null | undefined, currency = 'INR') => v == null ? 'Unavailable' : new Intl.NumberFormat(currency === 'INR' ? 'en-IN' : 'en-US', { style: 'currency', currency, maximumFractionDigits: 2 }).format(v);
export const timestamp = (v: string | null | undefined) => v ? new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }).format(new Date(v)) + ' UTC' : 'Unavailable';
export const label = (v: string) => v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
export function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
