import { describe, expect, it, vi } from 'vitest';
import { ApiError, makeApi } from '@/lib/api';

function jsonRes(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

describe('api client', () => {
  it('POST /retrieve sends JSON and returns the parsed body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonRes(200, { query: 'q', results: [] }));
    const api = makeApi(fetchMock as unknown as typeof fetch);

    const res = await api.retrieve({ query: 'q' });

    expect(res).toEqual({ query: 'q', results: [] });
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/retrieve$/);
    expect(init.method).toBe('POST');
    expect(init.headers['content-type']).toBe('application/json');
    expect(JSON.parse(init.body)).toEqual({ query: 'q' });
  });

  it('POST /resume URL-encodes the thread_id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonRes(200, {
      thread_id: 't/1', final: 'ok', trace: [], history: [],
    }));
    const api = makeApi(fetchMock as unknown as typeof fetch);

    await api.resume('t/1', { draft: 'edited' });

    const [url] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/resume\/t%2F1$/);
  });

  it('non-2xx throws ApiError with the parsed detail', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonRes(400, { detail: 'ann_k must be >= top_k' }));
    const api = makeApi(fetchMock as unknown as typeof fetch);

    await expect(api.retrieve({ query: 'q' })).rejects.toMatchObject({
      status: 400,
      name: 'ApiError',
      detail: { detail: 'ann_k must be >= top_k' },
    });
  });

  it('healthz uses GET', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonRes(200, {
      ok: true,
      qdrant: { ok: true, detail: null },
      neo4j:  { ok: true, detail: null },
      ollama: { ok: true, detail: null },
    }));
    const api = makeApi(fetchMock as unknown as typeof fetch);

    const h = await api.healthz();
    expect(h.ok).toBe(true);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('GET');
  });

  it('ApiError can be detected with instanceof', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonRes(500, {}));
    const api = makeApi(fetchMock as unknown as typeof fetch);

    try {
      await api.query({ query: 'q', thread_id: 't' });
      expect.fail('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
    }
  });
});
