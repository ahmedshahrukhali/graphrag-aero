import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryForm } from '@/components/QueryForm';

describe('QueryForm', () => {
  it('submits trimmed query with default filters as null', async () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText(/fuel/i), '  fuel exhaustion  ');
    await user.click(screen.getByRole('button', { name: /ask agent/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      query: 'fuel exhaustion',
      lang: null,
      source: null,
      maxHops: 2,
    });
  });

  it('propagates lang and source selections', async () => {
    const onSubmit = vi.fn();
    render(<QueryForm onSubmit={onSubmit} />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText(/fuel/i), 'carburant');
    await user.selectOptions(screen.getByLabelText(/language/i), 'fr');
    await user.selectOptions(screen.getByLabelText(/source/i), 'tsb');
    await user.click(screen.getByRole('button', { name: /ask agent/i }));

    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      lang: 'fr', source: 'tsb',
    }));
  });

  it('button is disabled until the query has content', async () => {
    render(<QueryForm onSubmit={vi.fn()} />);
    const btn = screen.getByRole('button', { name: /ask agent/i });
    expect(btn).toBeDisabled();

    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/fuel/i), 'x');
    expect(btn).not.toBeDisabled();
  });

  it('shows "Asking…" label when busy', () => {
    render(<QueryForm onSubmit={vi.fn()} busy />);
    expect(screen.getByRole('button', { name: /asking/i })).toBeInTheDocument();
  });
});
