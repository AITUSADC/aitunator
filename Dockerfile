# Stage 1: Build Vite + React + TypeScript frontend into static files
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/tsconfig.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src
RUN npm install --no-audit --no-fund && npm run build

# Stage 2: CPU Python 3.12 runtime with FastAPI + SQLite + RAG (torch CPU wheel index per TZ Section 7)
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    SQLITE_PATH=/app/runtime/aitunator.db \
    FRONTEND_DIST_DIR=/app/frontend/dist

WORKDIR /app

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY eval ./eval
COPY docs ./docs

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e ".[models,qdrant]" \
    && mkdir -p /app/runtime

COPY --from=frontend-builder /frontend/dist /app/frontend/dist

EXPOSE 8000

# Single worker replica required by SQLite WAL architecture (TZ Section 2)
CMD ["uvicorn", "backend.app.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
