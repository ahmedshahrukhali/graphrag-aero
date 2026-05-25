# P7 frontend image. Multi-stage so the runtime image only carries Next's
# standalone bundle + node_modules required at runtime, not the full
# devDependencies. Next is configured with `output: 'standalone'` so the
# build produces the minimal server bundle for us.

# ─── deps ────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# ─── build ───────────────────────────────────────────────────────────────────
FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
# NEXT_PUBLIC_* values get inlined at build time. Compose can override at run
# time only for *server-side* env reads; the browser bundle bakes in whatever
# this file resolves to.
ARG NEXT_PUBLIC_BACKEND_URL=http://localhost:8080
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL
RUN npm run build

# ─── runtime ────────────────────────────────────────────────────────────────
FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Standalone bundle includes the minimal server + node_modules.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
