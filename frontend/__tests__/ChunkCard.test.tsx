import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChunkCard } from '@/components/ChunkCard';
import type { RetrievedChunk } from '@/lib/types';

const baseChunk: RetrievedChunk = {
  rank: 1,
  doc_id: 'tsb/a00a0051',
  source_url: 'https://example.test/a00a0051.pdf',
  section_title: 'Findings as to causes',
  page: 4,
  bbox: [40, 320, 540, 400],
  lang: 'en',
  text: 'The flight crew elected to continue the visual approach\n  despite reduced visibility.',
  ann_score: 0.812,
  rerank_score: 0.974,
};

describe('ChunkCard', () => {
  it('shows rank, doc_id, page, section, snippet and scores', () => {
    render(<ChunkCard chunk={baseChunk} />);
    expect(screen.getByText(/#1/)).toBeInTheDocument();
    expect(screen.getByText(/tsb\/a00a0051/)).toBeInTheDocument();
    expect(screen.getByText(/p\.4/)).toBeInTheDocument();
    expect(screen.getByText(/Findings as to causes/)).toBeInTheDocument();
    expect(screen.getByText(/rerank=0\.974/)).toBeInTheDocument();
    expect(screen.getByText(/visual approach/)).toBeInTheDocument();
  });

  it('shows em-dash when rerank_score is null', () => {
    render(<ChunkCard chunk={{ ...baseChunk, rerank_score: null }} />);
    expect(screen.getByText(/rerank=—/)).toBeInTheDocument();
  });

  it('exposes a preview button only when onPreview is provided', async () => {
    const onPreview = vi.fn();
    render(<ChunkCard chunk={baseChunk} onPreview={onPreview} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /highlight in pdf/i }));
    expect(onPreview).toHaveBeenCalledWith(baseChunk);
  });
});
