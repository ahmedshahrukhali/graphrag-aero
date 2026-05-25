'use client';

interface Props {
  final: string | null;
}

export function FinalAnswer({ final }: Props) {
  if (!final) return null;
  return (
    <section className="space-y-1" aria-label="final-answer">
      <h3 className="text-sm font-semibold">Final answer</h3>
      <div className="rounded border-2 border-emerald-300 bg-emerald-50 p-3 font-serif text-sm leading-relaxed text-slate-900">
        {final}
      </div>
    </section>
  );
}
