// Typed client for the FastAPI backend.
//
// Configured via NEXT_PUBLIC_BACKEND_URL (defaults to http://localhost:8080).
// Every method throws ApiError on non-2xx so callers can surface the detail.

import type {
  HealthResponse,
  QueryPausedResponse,
  QueryRequest,
  ResumeRequest,
  ResumeResponse,
  RetrieveRequest,
  RetrieveResponse,
} from './types';

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
    this.name = 'ApiError';
  }
}

const DEFAULT_BASE_URL = 'http://localhost:8080';

export function getBaseUrl(): string {
  if (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_BACKEND_URL) {
    return process.env.NEXT_PUBLIC_BACKEND_URL;
  }
  return DEFAULT_BASE_URL;
}

async function request<T>(
  path: string,
  init: RequestInit,
  fetchImpl: typeof fetch = fetch,
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetchImpl(url, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init.headers ?? {}) },
  });
  if (!res.ok) {
    let detail: unknown = undefined;
    try { detail = await res.json(); } catch { /* body wasn't json */ }
    throw new ApiError(res.status, `${init.method ?? 'GET'} ${path} → ${res.status}`, detail);
  }
  return res.json() as Promise<T>;
}

export interface ApiClient {
  retrieve(req: RetrieveRequest): Promise<RetrieveResponse>;
  query(req: QueryRequest): Promise<QueryPausedResponse>;
  resume(threadId: string, req: ResumeRequest): Promise<ResumeResponse>;
  healthz(): Promise<HealthResponse>;
}

export function makeApi(fetchImpl: typeof fetch = fetch): ApiClient {
  return {
    retrieve: (req) =>
      request<RetrieveResponse>('/retrieve', { method: 'POST', body: JSON.stringify(req) }, fetchImpl),
    query: (req) =>
      request<QueryPausedResponse>('/query', { method: 'POST', body: JSON.stringify(req) }, fetchImpl),
    resume: (threadId, req) =>
      request<ResumeResponse>(
        `/resume/${encodeURIComponent(threadId)}`,
        { method: 'POST', body: JSON.stringify(req) },
        fetchImpl,
      ),
    healthz: () =>
      request<HealthResponse>('/healthz', { method: 'GET' }, fetchImpl),
  };
}

export const api = makeApi();
