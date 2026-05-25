import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AgentTrace } from '@/components/AgentTrace';

describe('AgentTrace', () => {
  it('renders an empty-state when no steps', () => {
    render(<AgentTrace trace={[]} />);
    expect(screen.getByText(/no trace/i)).toBeInTheDocument();
  });

  it('renders nodes in order with timing and extra fields', () => {
    render(
      <AgentTrace
        trace={[
          { node: 'retrieve',    elapsed_ms: 120, n_new: 7 },
          { node: 'graph_expand', elapsed_ms: 30,  n_ids: 4 },
          { node: 'synthesize',  elapsed_ms: 950, prompt_chars: 2103 },
        ]}
      />,
    );
    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent('retrieve');
    expect(items[0]).toHaveTextContent('120 ms');
    expect(items[0]).toHaveTextContent('n_new=7');
    expect(items[2]).toHaveTextContent('synthesize');
    expect(items[2]).toHaveTextContent('prompt_chars=2103');
  });
});
