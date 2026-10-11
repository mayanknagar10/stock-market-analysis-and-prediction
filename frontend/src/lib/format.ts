export const number = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? 'Unavailable' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(v);
export const percent = (v: number | null | undefined) => v == null ? 'Unavailable' : number(v * 100) + '%';
export const money = (v: number | null | undefined, currency = 'INR') => v == null ? 'Unavailable' : new Intl.NumberFormat(currency === 'INR' ? 'en-IN' : 'en-US', { style: 'currency', currency, maximumFractionDigits: 2 }).format(v);
export const compactMoney = (v: number | null | undefined, currency = 'INR') => {
    if (v == null || !Number.isFinite(v)) return 'Unavailable';
    const size = Math.abs(v);
    const [scale, suffix] = size >= 1e12 ? [1e12, 'T'] : size >= 1e9 ? [1e9, 'B'] : size >= 1e6 ? [1e6, 'M'] : [1, ''];
    return money(v / scale, currency) + suffix;
};
export const timestamp = (v: string | null | undefined) => v ? new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }).format(new Date(v)) + ' UTC' : 'Unavailable';
export const label = (v: string) => v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
export function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
