import './globals.css';
import type { ReactNode } from 'react';
import { HealthBadge } from '@/components/HealthBadge';

export const metadata = {
  title: 'GraphRAG Aero',
  description: 'Graph RAG over Transport Canada + TSB documents',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
          <div>
            <h1 className="text-base font-semibold">GraphRAG Aero</h1>
            <p className="text-xs text-slate-500">
              Graph RAG over Transport Canada + TSB documents
            </p>
          </div>
          <HealthBadge />
        </header>
        <main className="mx-auto max-w-5xl px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
