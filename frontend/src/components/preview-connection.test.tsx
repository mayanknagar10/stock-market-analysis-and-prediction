import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { PreviewConnection } from './preview-connection';

beforeEach(() => localStorage.clear());

it('accepts only HTTPS temporary backend URLs and keeps the choice in this browser', () => {
    render(<PreviewConnection/>);
    fireEvent.change(screen.getByLabelText('Temporary HTTPS tunnel URL'), { target: { value: 'http://insecure.example' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save backend URL' }));
    expect(screen.getByRole('alert')).toHaveTextContent('HTTPS');
    expect(localStorage.getItem('stockpro.previewApiBaseUrl')).toBeNull();

    fireEvent.change(screen.getByLabelText('Temporary HTTPS tunnel URL'), { target: { value: 'https://new-tunnel.trycloudflare.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save backend URL' }));
    expect(localStorage.getItem('stockpro.previewApiBaseUrl')).toBe('https://new-tunnel.trycloudflare.com');
    expect(screen.getByRole('link', { name: 'Reload to connect' })).toBeInTheDocument();
});
