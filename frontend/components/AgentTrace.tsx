'use client';

import type { TraceStep } from '@/lib/types';

interface Props {
  trace: TraceStep[];
}

// Render the per-node timeline returned by /query and /resume. Each step is
// {node, elapsed_ms, ...}; we surface the node name + timing + any extra keys
// so the user can audit what the agent did at every hop.
export function AgentTrace({ trace }: Props) {
  if (trace.length === 0) {
    return <p className="text-sm text-slate-500">No trace yet.</p>;
  }
  return (
    <ol className="space-y-2" aria-label="agent-trace">
      {trace.map((step, i) => {
        const { node, elapsed_ms, ...rest } = step;
        const extras = Object.entries(rest)
          .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
          .join('  ');
        return (
          <li key={i} className="rounded border border-slate-200 bg-slate-50 p-2 text-sm">
            <div className="flex items-baseline justify-between">
              <span className="font-mono font-semibold">{node}</span>
              <span className="text-xs text-slate-500">{elapsed_ms} ms</span>
            </div>
            {extras && <div className="font-mono text-xs text-slate-600">{extras}</div>}
          </li>
        );
      })}
    </ol>
  );
}
