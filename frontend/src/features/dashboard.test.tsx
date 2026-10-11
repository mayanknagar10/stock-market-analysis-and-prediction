import { describe, expect, it } from 'vitest';
import { dataHealthColor } from './dashboard';

describe('authoritative collection health indicator', () => {
    it.each(['HEALTHY', 'OK'])('shows confirmed %s health as positive', health => {
        expect(dataHealthColor(health)).toBe('var(--positive)');
    });
    it.each([undefined, 'UNKNOWN', 'UNAVAILABLE', 'NOT_STARTED'])('shows %s without a success indicator', health => {
        expect(dataHealthColor(health)).toBe('var(--muted)');
    });
    it.each(['DEGRADED', 'WARNING', 'FAILED', 'PARTIAL'])('warns for %s collection health', health => {
        expect(dataHealthColor(health)).toBe('var(--warning)');
    });
});
