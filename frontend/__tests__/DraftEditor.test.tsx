import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DraftEditor } from '@/components/DraftEditor';

describe('DraftEditor', () => {
  it('reports edited=false when the user clicks finalize without changes', async () => {
    const onFinalize = vi.fn();
    render(<DraftEditor initial="model draft" onFinalize={onFinalize} />);
    await userEvent.setup().click(screen.getByRole('button', { name: /finalize/i }));
    expect(onFinalize).toHaveBeenCalledWith('model draft', false);
  });

  it('reports edited=true and the new text when the user types', async () => {
    const onFinalize = vi.fn();
    render(<DraftEditor initial="model draft" onFinalize={onFinalize} />);
    const user = userEvent.setup();
    const ta = screen.getByRole('textbox');
    await user.clear(ta);
    await user.type(ta, 'human edit');
    await user.click(screen.getByRole('button', { name: /submit edit/i }));
    expect(onFinalize).toHaveBeenCalledWith('human edit', true);
  });
});
