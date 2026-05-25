'use client';

import { useState } from 'react';
import { api, ApiError } from '@/lib/api';
import { newThreadId } from '@/lib/threadId';
import type {
  QueryPausedResponse,
  ResumeResponse,
  RetrievedChunk,
  RetrieveResponse,
} from '@/lib/types';
import { QueryForm, type QuerySubmission } from '@/components/QueryForm';
import { AgentTrace } from '@/components/AgentTrace';
import { ChunkCard } from '@/components/ChunkCard';
import { DraftEditor } from '@/components/DraftEditor';
import { FinalAnswer } from '@/components/FinalAnswer';
import { PdfPreview } from '@/components/PdfPreview';

// Top-level state machine for one HITL session.
type Stage =
  | { kind: 'idle' }
  | { kind: 'asking' }
  | { kind: 'paused';   query: string; paused: QueryPausedResponse; retrieve: RetrieveResponse }
  | { kind: 'resuming'; query: string; paused: QueryPausedResponse; retrieve: RetrieveResponse }
  | { kind: 'done';     query: string; paused: QueryPausedResponse; retrieve: RetrieveResponse; resume: ResumeResponse };

export default function Home() {
  const [stage, setStage] = useState<Stage>({ kind: 'idle' });
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<RetrievedChunk | null>(null);

  const ask = async (s: QuerySubmission) => {
    setError(null);
    setStage({ kind: 'asking' });
    const threadId = newThreadId();
    try {
      // Run /retrieve and /query in parallel — /query also retrieves
      // internally, but the agent merges hops and we want the user to see the
      // candidate set the agent saw, not the dedupped hop merge.
      const [retrieve, paused] = await Promise.all([
        api.retrieve({ query: s.query, lang: s.lang, source: s.source, top_k: 10 }),
        api.query({ query: s.query, thread_id: threadId, max_hops: s.maxHops }),
      ]);
      setStage({ kind: 'paused', query: s.query, paused, retrieve });
    } catch (e) {
      setError(formatError(e));
      setStage({ kind: 'idle' });
    }
  };

  const finalize = async (draft: string, edited: boolean) => {
    if (stage.kind !== 'paused') return;
    const prev = stage;
    setStage({ ...prev, kind: 'resuming' });
    try {
      const resume = await api.resume(prev.paused.thread_id, edited ? { draft } : {});
      setStage({ kind: 'done', query: prev.query, paused: prev.paused, retrieve: prev.retrieve, resume });
    } catch (e) {
      setError(formatError(e));
      setStage(prev);
    }
  };

  return (
    <div className="space-y-6">
      <QueryForm onSubmit={ask} busy={stage.kind === 'asking'} />
      {error && (
        <div role="alert" className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {(stage.kind === 'paused' || stage.kind === 'resuming' || stage.kind === 'done') && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_22rem]">
          <div className="space-y-4">
            {stage.kind === 'done' && stage.resume.final && (
              <FinalAnswer final={stage.resume.final} />
            )}
            {(stage.kind === 'paused' || stage.kind === 'resuming') && (
              <section className="space-y-1">
                <h3 className="text-sm font-semibold">Draft (HITL gate)</h3>
                <DraftEditor
                  key={stage.paused.thread_id}
                  initial={stage.paused.draft ?? ''}
                  busy={stage.kind === 'resuming'}
                  onFinalize={finalize}
                />
              </section>
            )}
            <section className="space-y-2">
              <h3 className="text-sm font-semibold">Cited chunks ({stage.retrieve.results.length})</h3>
              {stage.retrieve.results.map((c) => (
                <ChunkCard key={c.rank} chunk={c} onPreview={setPreview} />
              ))}
            </section>
          </div>
          <aside className="space-y-4">
            <section className="space-y-1">
              <h3 className="text-sm font-semibold">Agent trace</h3>
              <AgentTrace
                trace={stage.kind === 'done' ? stage.resume.trace : stage.paused.trace}
              />
            </section>
            <p className="text-xs text-slate-500">thread_id: <span className="font-mono">{stage.paused.thread_id}</span></p>
          </aside>
        </div>
      )}

      {preview && <PdfPreview chunk={preview} onClose={() => setPreview(null)} />}
    </div>
  );
}

function formatError(e: unknown): string {
  if (e instanceof ApiError) {
    const detail = typeof e.detail === 'object' && e.detail && 'detail' in e.detail
      ? String((e.detail as { detail: unknown }).detail)
      : '';
    return detail ? `${e.message} — ${detail}` : e.message;
  }
  return e instanceof Error ? e.message : String(e);
}
