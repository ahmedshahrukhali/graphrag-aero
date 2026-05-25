# frontend/ — P7

Next.js 14 (App Router) + TypeScript UI for the GraphRAG Aero backend.

## What it does
- POSTs the question to the backend's `/query` endpoint and renders the
  paused HITL state: agent trace, retrieved chunks, and the draft.
- Lets the user edit the draft and POST it back to `/resume/{thread_id}`
  to get the final answer.
- Each chunk card has a "highlight in pdf" action that opens a modal
  with the source PDF (`react-pdf` + PDF.js) and overlays a coloured box
  positioned from the chunk's `bbox` metadata (pdfplumber coords).
- `/healthz` is polled every 30s; the result shows as a pill in the
  header with per-component status on hover.

## Stack
- Next.js 14 (`output: 'standalone'`)
- React 18, TypeScript 5
- TailwindCSS for styling
- `react-pdf` for PDF rendering
- Vitest + Testing Library for tests

## Configure
```
cp frontend/.env.local.example frontend/.env.local
# edit NEXT_PUBLIC_BACKEND_URL if the backend isn't on http://localhost:8080
```
`NEXT_PUBLIC_BACKEND_URL` is inlined at build time. When running the
container, pass it as a Docker build-arg (compose does this for you).

## Dev
```bash
cd frontend
npm install
npm run dev          # → http://localhost:3000
npm run test         # vitest
npm run typecheck    # tsc --noEmit
npm run build        # production build (also runs the type-checker)
```

## Docker
```bash
docker compose up --build frontend
```
The multi-stage `Dockerfile` produces a slim runtime image with the
`output: 'standalone'` bundle. The build arg `NEXT_PUBLIC_BACKEND_URL`
gets baked into the browser bundle; override it in compose with a real
backend URL if you're running the stack behind a reverse proxy.

## Layout
- [app/](app/) — App Router root layout + the single-page UI
- [components/](components/) — `QueryForm`, `AgentTrace`, `ChunkCard`,
  `DraftEditor`, `FinalAnswer`, `HealthBadge`, `PdfPreview`
- [lib/](lib/) — typed API client + TS mirrors of the backend Pydantic
  schemas + `newThreadId()`
- [__tests__/](__tests__/) — Vitest specs (api, components, bbox math)

## Out of scope (deferred)
- Auth — single-user local stack
- Streaming responses — backend returns full paused-state today
- A separate `/retrieve` debug page — the main page already shows the
  retrieve hits next to the agent state
