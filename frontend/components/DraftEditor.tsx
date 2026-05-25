'use client';

import { useEffect, useState } from 'react';

interface Props {
  initial: string;
  busy?: boolean;
  onFinalize: (draft: string, edited: boolean) => void;
}

// The HITL gate. Shows the model's draft and lets the user edit it before
// finalizing. ``edited`` is forwarded to the parent so it can decide whether
// to send a ``draft`` field in the /resume body (sending the original draft
// unchanged is wasteful and confusing in audit logs).
export function DraftEditor({ initial, busy = false, onFinalize }: Props) {
  const [draft, setDraft] = useState(initial);

  useEffect(() => {
    setDraft(initial);
  }, [initial]);

  const edited = draft !== initial;

  return (
    <div className="space-y-2" aria-label="draft-editor">
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        rows={10}
        className="w-full rounded border border-slate-300 p-2 font-serif text-sm"
      />
      <div className="flex items-center gap-3 text-sm">
        <button
          type="button"
          disabled={busy}
          onClick={() => onFinalize(draft, edited)}
          className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {busy ? 'Finalizing…' : edited ? 'Submit edit & finalize' : 'Finalize'}
        </button>
        {edited && <span className="text-xs text-amber-700">draft modified</span>}
      </div>
    </div>
  );
}
