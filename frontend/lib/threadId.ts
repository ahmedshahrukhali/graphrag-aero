// Generates a thread_id for a new HITL session. Uses crypto.randomUUID where
// available (modern browsers + Node 19+), falls back to a Math.random ID.
export function newThreadId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  return `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
