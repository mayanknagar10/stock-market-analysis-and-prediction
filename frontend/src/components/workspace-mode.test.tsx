import { render, screen, fireEvent } from '@testing-library/react';
import { StrictMode } from 'react';
import { beforeEach, describe, it, expect } from 'vitest';
import { WorkspaceMode, ModeSwitch, AdvancedDetails } from './workspace-mode';
function Workspace() { return <StrictMode><WorkspaceMode><ModeSwitch/><AdvancedDetails title="Saved diagnostics"><p>Archived values</p></AdvancedDetails></WorkspaceMode></StrictMode>; }
beforeEach(() => localStorage.clear());
describe('workspace detail preference', () => {
    it('lets users reveal diagnostics and persists their selected detail level', () => { render(<Workspace/>); expect(screen.getByText('Saved diagnostics').closest('details')).not.toHaveAttribute('open'); fireEvent.click(screen.getByRole('button', { name: 'Advanced' })); expect(screen.getByText('Saved diagnostics').closest('details')).toHaveAttribute('open'); expect(localStorage.getItem('stockpro.mode')).toBe('advanced'); fireEvent.click(screen.getByRole('button', { name: 'Simple' })); expect(screen.getByText('Saved diagnostics').closest('details')).not.toHaveAttribute('open'); });
    it('restores the advanced preference under StrictMode', () => { localStorage.setItem('stockpro.mode', 'advanced'); render(<Workspace/>); expect(screen.getByRole('button', { name: 'Advanced' })).toHaveAttribute('aria-pressed', 'true'); expect(localStorage.getItem('stockpro.mode')).toBe('advanced'); });
});
