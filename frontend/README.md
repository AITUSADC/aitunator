# Веб-интерфейс AITUNATOR (Vite + React + TypeScript)

Реализовано по ТЗ и `docs/contract.md`:
- Потоковый чат (`POST /api/chat` по SSE) с индикацией этапов (`planning`, `retrieving`, `generating`) и кнопкой **«Стоп»** (`AbortController`).
- Восстановление истории при перезагрузке страницы (`GET /api/sessions/{id}`), в `localStorage` хранится только `session_id`.
- Выбор образовательной программы (`GET /api/programs`) и карточки с примерами вопросов.
- Оценка ответов (`POST /api/feedback`) с необязательным комментарием (скрывается для ответов с `message_id: null`).
- Безопасный рендеринг Markdown (`src/markdown.tsx`) без сырого HTML и без показа ссылок на источники.
- Адаптивная вёрстка от ширины `360px`.

## Разработка и сборка

```bash
cd frontend
npm install
npm run dev      # dev-сервер с прокси /api -> http://127.0.0.1:8000
npm run build    # сборка статики в frontend/dist для раздачи через FastAPI
```
