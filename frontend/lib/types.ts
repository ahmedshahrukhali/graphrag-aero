// TypeScript mirrors of the backend Pydantic schemas (see backend/schemas.py).
// Keep these in lock-step with the backend; the API client uses them as the
// contract for every request/response.

export type Lang = 'en' | 'fr';
export type Source = 'tsb' | 'tc';

export interface RetrieveRequest {
  query: string;
  lang?: Lang | null;
  source?: Source | null;
  ann_k?: number;
  top_k?: number;
}

export interface RetrievedChunk {
  rank: number;
  doc_id: string;
  source_url: string | null;
  section_title: string;
  page: number;
  bbox: [number, number, number, number];
  lang: string;
  text: string;
  ann_score: number;
  rerank_score: number | null;
}

export interface RetrieveResponse {
  query: string;
  results: RetrievedChunk[];
}

// ---- /query + /resume ----

export interface QueryRequest {
  query: string;
  thread_id: string;
  max_hops?: number;
}

export interface TraceStep {
  node: string;
  elapsed_ms: number;
  [extra: string]: unknown;
}

export interface QueryPausedResponse {
  thread_id: string;
  draft: string | null;
  trace: TraceStep[];
  n_candidates: number;
}

export interface ResumeRequest {
  draft?: string | null;
}

export interface HistoryStep {
  step: number;
  next: string[];
  values_summary: {
    hop: number | null;
    n_candidates: number;
    draft_present: boolean;
    final_present: boolean;
  };
}

export interface ResumeResponse {
  thread_id: string;
  final: string | null;
  trace: TraceStep[];
  history: HistoryStep[];
}

// ---- /healthz ----

export interface ComponentHealth {
  ok: boolean;
  detail: string | null;
}

export interface HealthResponse {
  ok: boolean;
  qdrant: ComponentHealth;
  neo4j: ComponentHealth;
  ollama: ComponentHealth;
}
