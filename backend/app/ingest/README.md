# Подготовка базы

`crawl.py` — сбор `data/md` с astanait.edu.kz: `python -m backend.app.ingest.crawl` (нужен extra `crawl`).
Страницы и целевые файлы заданы в `SOURCES`; HTML кэшируется в `runtime/raw_html`.
`documents.py` — документы Google Drive/Docs со страниц: отбор ссылок (без списков с ПДн, договоров,
сертификатов и сканов учебных планов; вырезаются таблицы с колонкой ФИО и страницы с подписями), загрузка в `runtime/raw_docs`,
PDF через pymupdf4llm, DOCX через mammoth. Нечитаемые страницы распознаются Tesseract
(`apt install tesseract-ocr tesseract-ocr-rus`) с автоповоротом по OSD; сырой Markdown кэшируется в `{id}.md`,
поэтому правки фильтров ПДн не требуют повторного OCR.
Для обновления удалите `runtime/raw_html` и `runtime/raw_docs`.
`extract.py` — подготовка чанков из Markdown и опциональный старый PDF-парсер.
`index.py` — индексация: `python -m backend.app.ingest.index --index-md-dir data/md`.
Загрузка Drive идёт через `/uc`, который запрещён robots.txt Google; это осознанное решение проекта (1 запрос/с).
Ревью через PR в aitu-kb и смена alias пока не реализованы.
