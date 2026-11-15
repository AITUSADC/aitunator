# Архитектура

Один сервис FastAPI (API + RAG) и Qdrant. Фронтенд собирается в статику и отдаётся тем же сервисом.

## Модули

| Модуль | Назначение | Зависит от |
|---|---|---|
| `backend/app/api` | HTTP и SSE (`/api/chat`, `sessions`, `feedback`, `programs`, `health`), SQLite, rate limit, фильтр ПДн | `rag.pipeline`, `rag.programs` |
| `backend/app/rag` | Поиск (BM25 + эмбеддинги, RRF, разбиение по программам, каталог для вопросов о списке программ), генерация: ответ по фактам или консультация с уточняющими вопросами | `providers`, локальный корпус |
| `backend/app/providers` | LLM (OpenAI-compatible chat completions по httpx); эмбеддер: локальный Sentence Transformers или `openai/<model>` через `/embeddings` | Groq / совместимый endpoint, OpenAI |
| `backend/app/ingest` | `crawl` — сайт → `data/md` и каталог `programs/index.md` из реестра; `documents` — PDF/DOCX из Google Drive (OCR Tesseract); `extract` — Markdown → чанки; `index` — чанки → `runtime/points_cache.jsonl` и Qdrant | astanait.edu.kz, Google Drive, Tesseract, BGE-M3, Qdrant |
| `eval` | Офлайн-оценка поиска по `golden.jsonl` | `rag`, `runtime/points_cache.jsonl` |
| `frontend` | Vite + React + TypeScript | `/api/*` |

## Данные

```text
astanait.edu.kz ──crawl──▶ runtime/raw_html (кэш) ──▶ data/md/*.md (frontmatter + ## разделы)
ссылки на Drive ──documents──▶ runtime/raw_docs (кэш) ──PDF/DOCX/OCR──▶ data/md (тот же файл или по маршруту)
data/md ──index──▶ runtime/points_cache.jsonl ──▶ Qdrant (коллекция QDRANT_COLLECTION)
points_cache.jsonl ──▶ runtime: BM25 + точный cosine в памяти (Qdrant в runtime пока не используется)
```

Имена файлов в `data/md` совпадают с реестром `rag/programs.py`: по ним выводятся программа и тип документа.
`data/`, `runtime/` и `tmp/models` не входят в Git.

## Инфраструктура

- `docker-compose.yml`: `app` (одна реплика из-за SQLite WAL) и `qdrant`.
- SQLite: `runtime/aitunator.db` — сессии, сообщения, оценки.
- Модели: `EMBEDDING_MODEL` — BGE-M3 в `MODEL_CACHE_DIR` (`tmp/models`) или `openai/text-embedding-3-large` (1024 dims, `OPENAI_API_KEY`);
  индекс и запросы обязаны использовать одну модель. LLM через `LLM_PROVIDER`.
