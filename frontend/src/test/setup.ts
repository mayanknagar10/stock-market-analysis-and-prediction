import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';
afterEach(cleanup);
class ResizeObserverMock {
    observe() { }
    unobserve() { }
    disconnect() { }
}
vi.stubGlobal('ResizeObserver', ResizeObserverMock);
Object.defineProperty(window, 'matchMedia', { writable: true, value: vi.fn().mockImplementation(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })) });
