'use client';

import { FormEvent, useState } from 'react';
import type { Lang, Source } from '@/lib/types';

export interface QuerySubmission {
  query: string;
  lang: Lang | null;
  source: Source | null;
  maxHops: number;
}

interface Props {
  onSubmit: (s: QuerySubmission) => void;
  busy?: boolean;
}

export function QueryForm({ onSubmit, busy = false }: Props) {
  const [query, setQuery] = useState('');
  const [lang, setLang] = useState<'all' | Lang>('all');
  const [source, setSource] = useState<'all' | Source>('all');
  const [maxHops, setMaxHops] = useState(2);

  const handle = (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit({
      query: query.trim(),
      lang: lang === 'all' ? null : lang,
      source: source === 'all' ? null : source,
      maxHops,
    });
  };

  return (
    <form onSubmit={handle} className="space-y-3" aria-label="query-form">
      <label className="block">
        <span className="block text-sm font-medium">Question</span>
        <textarea
          name="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={3}
          className="mt-1 w-full rounded border border-slate-300 p-2 font-mono text-sm"
          placeholder="e.g. fuel exhaustion forced landing"
        />
      </label>
      <div className="flex flex-wrap gap-3">
        <label>
          <span className="block text-sm">Language</span>
          <select
            value={lang}
            onChange={(e) => setLang(e.target.value as 'all' | Lang)}
            className="rounded border border-slate-300 p-1"
          >
            <option value="all">all</option>
            <option value="en">en</option>
            <option value="fr">fr</option>
          </select>
        </label>
        <label>
          <span className="block text-sm">Source</span>
          <select
            value={source}
            onChange={(e) => setSource(e.target.value as 'all' | Source)}
            className="rounded border border-slate-300 p-1"
          >
            <option value="all">all</option>
            <option value="tsb">tsb</option>
            <option value="tc">tc</option>
          </select>
        </label>
        <label>
          <span className="block text-sm">Max hops</span>
          <input
            type="number"
            min={1}
            max={5}
            value={maxHops}
            onChange={(e) => setMaxHops(Number(e.target.value))}
            className="w-20 rounded border border-slate-300 p-1"
          />
        </label>
      </div>
      <button
        type="submit"
        disabled={busy || !query.trim()}
        className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
      >
        {busy ? 'Asking…' : 'Ask agent'}
      </button>
    </form>
  );
}
