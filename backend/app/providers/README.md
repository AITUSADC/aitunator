# Провайдеры

`embeddings/sentence_transformer.py` содержит готовые вызовы эмбеддера.
`embeddings/openai.py` — эмбеддер OpenAI-compatible `/embeddings` с тем же `.encode()`;
включается `EMBEDDING_MODEL=openai/<model>` (ключ `OPENAI_API_KEY`, `OPENAI_BASE_URL` опционально).
`llm/chat.py` содержит Protocol, настройки и асинхронный chat-completions адаптер
для Groq или явно настроенного совместимого endpoint. Основная модель выдаёт
SSE-поток, быстрая — JSON для обработки истории и проверки контекста.
Клиент создаётся при startup и закрывается при shutdown.
`rerank/` остаётся зарезервированным: runtime использует существующее ранжирование
RRF и приоритет разделов без нейросетевого reranker.
