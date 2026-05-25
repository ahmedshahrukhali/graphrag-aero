'use client';

import { useEffect, useMemo, useState } from 'react';
import type { RetrievedChunk } from '@/lib/types';

interface Props {
  chunk: RetrievedChunk;
  onClose: () => void;
}

// pdfplumber bbox convention: [x0, top, x1, bottom] in PDF points with the
// origin at the top-left of the page (pdfplumber returns top-origin coords by
// default). Convert to canvas-relative percentages so the overlay tracks the
// rendered page regardless of zoom.
export function bboxToOverlayStyle(
  bbox: [number, number, number, number],
  pageWidthPts: number,
  pageHeightPts: number,
): { left: string; top: string; width: string; height: string } {
  const [x0, top, x1, bottom] = bbox;
  const width = Math.max(0, x1 - x0);
  const height = Math.max(0, bottom - top);
  return {
    left:   `${(x0 / pageWidthPts) * 100}%`,
    top:    `${(top / pageHeightPts) * 100}%`,
    width:  `${(width / pageWidthPts) * 100}%`,
    height: `${(height / pageHeightPts) * 100}%`,
  };
}

// The react-pdf Document/Page components only work in the browser (PDF.js
// touches the DOM). We dynamic-import them on mount so the modal can still
// SSR-render its chrome — the canvas slot stays empty until client-side.
export function PdfPreview({ chunk, onClose }: Props) {
  const [Doc, setDoc] = useState<null | typeof import('react-pdf')>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    let active = true;
    import('react-pdf').then((mod) => {
      // Worker bootstrap — react-pdf needs the pdfjs worker URL set once.
      mod.pdfjs.GlobalWorkerOptions.workerSrc =
        `https://unpkg.com/pdfjs-dist@${mod.pdfjs.version}/build/pdf.worker.min.js`;
      if (active) setDoc(mod);
    });
    return () => { active = false; };
  }, []);

  const overlay = useMemo(() => {
    if (!size) return null;
    return bboxToOverlayStyle(chunk.bbox, size.width, size.height);
  }, [chunk.bbox, size]);

  if (!chunk.source_url) {
    return (
      <Modal onClose={onClose}>
        <p className="text-sm">No source URL attached to this chunk.</p>
      </Modal>
    );
  }

  return (
    <Modal onClose={onClose}>
      <header className="flex items-baseline justify-between">
        <span className="text-sm font-mono">
          {chunk.doc_id} · p.{chunk.page}
        </span>
        <button onClick={onClose} className="text-xs text-slate-500">close</button>
      </header>
      <div className="relative inline-block">
        {Doc ? (
          <Doc.Document file={chunk.source_url} loading="loading…">
            <Doc.Page
              pageNumber={chunk.page}
              renderTextLayer={false}
              renderAnnotationLayer={false}
              onLoadSuccess={(p) => setSize({ width: p.width, height: p.height })}
            />
          </Doc.Document>
        ) : (
          <div className="h-96 w-72 animate-pulse bg-slate-100" />
        )}
        {overlay && (
          <div
            aria-label="bbox-overlay"
            className="pointer-events-none absolute border-2 border-amber-500 bg-amber-200/30"
            style={overlay}
          />
        )}
      </div>
    </Modal>
  );
}

function Modal({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] overflow-auto rounded bg-white p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
