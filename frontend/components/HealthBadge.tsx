'use client';

import { useEffect, useState } from 'react';
import { api, ApiError } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

interface Props {
  pollMs?: number;
}

// Small status pill in the header. Polls /healthz and shows per-component
// status on hover via the ``title`` attribute (no separate popover — keep the
// component dependency-free).
export function HealthBadge({ pollMs = 30_000 }: Props) {
  const [state, setState] = useState<HealthResponse | { ok: false; error: string } | null>(null);

  useEffect(() => {
    let active = true;
    const tick = async () => {
      try {
        const h = await api.healthz();
        if (active) setState(h);
      } catch (e) {
        if (active) {
          const msg = e instanceof ApiError ? `${e.status}` : 'unreachable';
          setState({ ok: false, error: msg });
        }
      }
    };
    void tick();
    const id = window.setInterval(tick, pollMs);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [pollMs]);

  if (state === null) {
    return <span className="rounded bg-slate-200 px-2 py-0.5 text-xs">health …</span>;
  }

  if ('error' in state) {
    return (
      <span
        className="rounded bg-red-200 px-2 py-0.5 text-xs text-red-900"
        title={`backend unreachable: ${state.error}`}
      >
        backend down
      </span>
    );
  }

  const title = [
    `qdrant: ${state.qdrant.ok ? 'ok' : state.qdrant.detail ?? 'down'}`,
    `neo4j:  ${state.neo4j.ok ? 'ok' : state.neo4j.detail ?? 'down'}`,
    `ollama: ${state.ollama.ok ? 'ok' : state.ollama.detail ?? 'down'}`,
  ].join('\n');

  return (
    <span
      className={`rounded px-2 py-0.5 text-xs ${
        state.ok ? 'bg-emerald-200 text-emerald-900' : 'bg-amber-200 text-amber-900'
      }`}
      title={title}
    >
      {state.ok ? 'all systems ok' : 'degraded'}
    </span>
  );
}
