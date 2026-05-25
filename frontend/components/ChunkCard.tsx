'use client';

import type { RetrievedChunk } from '@/lib/types';

interface Props {
  chunk: RetrievedChunk;
  onPreview?: (c: RetrievedChunk) => void;
}

export function ChunkCard({ chunk, onPreview }: Props) {
  const rerank = chunk.rerank_score?.toFixed(3) ?? '—';
  const ann = chunk.ann_score.toFixed(3);
  const snippet = chunk.text.replace(/\s+/g, ' ').trim().slice(0, 280);

  return (
    <article className="space-y-1 rounded border border-slate-200 p-3" aria-label="chunk-card">
      <header className="flex items-baseline justify-between text-xs text-slate-500">
        <span className="font-mono">
          #{chunk.rank} · {chunk.doc_id} · p.{chunk.page}
        </span>
        <span>rerank={rerank} · ann={ann}</span>
      </header>
      {chunk.section_title && (
        <div className="text-sm font-medium">§ {chunk.section_title}</div>
      )}
      <p className="font-serif text-sm leading-snug text-slate-800">{snippet}…</p>
      <footer className="flex items-center gap-3 text-xs">
        <span className="rounded bg-slate-100 px-1.5 py-0.5">{chunk.lang}</span>
        {chunk.source_url && (
          <a
            href={chunk.source_url}
            target="_blank"
            rel="noreferrer"
            className="text-blue-700 underline"
          >
            source pdf
          </a>
        )}
        {onPreview && (
          <button
            type="button"
            className="text-blue-700 underline"
            onClick={() => onPreview(chunk)}
          >
            highlight in pdf
          </button>
        )}
      </footer>
    </article>
  );
}
