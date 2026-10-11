import logging
import re

from backend.app.config import CHUNK_OVERLAP, MAX_CHARS_PER_CHUNK
from backend.app.rag.programs import get_markdown_metadata, get_program_metadata

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEP = re.compile(r"\|[-: |]+\|")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_NOISY_TABLE_RE = re.compile(
    r"ТОО|Редакция|Протокол|ДП-AITU|Ученого совета|утвержден|подпись|согласован"
    r"|Мероприятия|KPI|ЗАДАЧИ|Наименование риска|Лист регистрации|Лист ознакомления"
    r"|ФИО|Должность|Ед\.изм\.|уч\.г\.|Штатных|Привлеченных|Кол-во|Параметры"
    r"|аудитори|Материально|Кадровое|SWOT|сильные стороны|слабые стороны",
    re.IGNORECASE,
)
_USEFUL_TABLE_RE = re.compile(
    r"кредит|час|ECTS|дисциплин|семестр|балл|компетенц|модуль|практик|лекци|итого"
    r"|триместр|обеспеченность|пререквизит|постреквизит",
    re.IGNORECASE,
)

_SECTION_TYPES = {
    "кратко": "overview",
    "краткое описание": "overview",
    "основная информация": "overview",
    "цель программы": "overview",
    "overview": "overview",
    "для кого": "fit",
    "чему научишься": "skills",
    "навыки": "skills",
    "ключевые результаты": "skills",
    "ключевые области": "skills",
    "области применения": "skills",
    "ключевые дисциплины": "curriculum",
    "дисциплины": "curriculum",
    "направления дисциплин": "curriculum",
    "основные дисциплины": "curriculum",
    "элективные дисциплины": "curriculum",
    "итоговая аттестация": "curriculum",
    "карьера": "career",
    "возможные позиции выпускника": "career",
    "трудоустройство": "career",
    "поступление": "admission",
    "направления поступления": "admission",
    "приемная комиссия": "admission",
    "faq": "faq",
    "частые вопросы": "faq",
}


def parse_with_docling(file_path: str) -> str:
    """Convert a PDF to Markdown when the optional pdf dependencies are installed."""
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "PDF indexing requires the optional dependencies: run `uv sync --extra pdf`."
        ) from exc

    options = PdfFormatOption(
        pipeline_options=PdfPipelineOptions(
            generate_page_images=False,
            generate_picture_images=False,
        )
    )
    converter = DocumentConverter(format_options={"pdf": options})
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",")]
    if value.lower() in {"null", "none"}:
        return None
    if value.isdigit():
        return int(value)
    return value.strip("\"'")


def _strip_frontmatter(text: str) -> tuple[str, dict]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text, {}

    metadata = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.strip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)
    return text[match.end():], metadata


def extract_sections_markdown(text: str) -> dict:
    """Split markdown into sections by ## headings. Returns {heading: content}."""
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    headings = list(pattern.finditer(text))
    if not headings:
        return {"document": text.strip()}

    sections = {}
    for i, match in enumerate(headings):
        title = match.group(1).strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        content = text[start:end].strip()
        if content:
            key = title if title not in sections else f"{title}_{i}"
            sections[key] = content
    return sections


def _is_markdown_table(text: str) -> bool:
    """Return True if text contains a proper Markdown table (has header-separator row)."""
    return "|" in text and bool(_TABLE_SEP.search(text))


def _is_noisy_table(text: str) -> bool:
    """Return True for administrative/header tables that add no retrieval value."""
    data_rows = [line for line in text.split("\n") if "|" in line and "---" not in line]
    if len(data_rows) <= 3:
        return True
    if _NOISY_TABLE_RE.search(text) and not _USEFUL_TABLE_RE.search(text):
        return True
    return False


def _infer_section_type(section_name: str) -> str:
    normalized = section_name.strip().lower()
    for prefix, section_type in _SECTION_TYPES.items():
        if normalized.startswith(prefix):
            return section_type
    return "general"


def _normalize_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _base_payload(
    *,
    source_name: str,
    section_name: str,
    source_format: str,
    source_group: str,
    doc_type: str,
    metadata: dict | None = None,
    chunk_index: int = 0,
    content_type: str = "text_part",
) -> dict:
    metadata = metadata or {}
    program_metadata = get_program_metadata(source_name)
    payload = {
        "section": section_name,
        "source": source_name,
        "source_title": metadata.get("source_title")
        or metadata.get("program_name")
        or program_metadata.get("program_name")
        or source_name,
        "source_format": metadata.get("source_format", source_format),
        "source_group": metadata.get("source_group", source_group),
        "doc_type": metadata.get("doc_type", doc_type),
        "language": metadata.get("language", "ru"),
        "program_code": metadata.get("program_code", program_metadata.get("program_code")),
        "program_slug": metadata.get("program_slug", program_metadata.get("program_slug")),
        "program_name": metadata.get("program_name", program_metadata.get("program_name")),
        "school": metadata.get("school", program_metadata.get("school")),
        "section_type": metadata.get("section_type") or _infer_section_type(section_name),
        "topic_tags": _normalize_list(metadata.get("topic_tags")),
        "audience": _normalize_list(metadata.get("audience")),
        "priority": int(metadata.get("priority", 90)),
        "chunk_index": chunk_index,
        "parent_id": f"{source_name}::{section_name}",
        "updated_at": metadata.get("updated_at"),
        "valid_for": metadata.get("valid_for"),
        "valid_from": metadata.get("valid_from"),
        "valid_until": metadata.get("valid_until"),
        "content_status": metadata.get("content_status"),
        "type": content_type,
    }
    return payload


def _embedding_prefix(payload: dict) -> str:
    parts = [
        f"[Документ: {payload.get('source_title') or payload.get('source')}]",
        f"[Тип: {payload.get('doc_type')}]",
        f"[Источник: {payload.get('source_group')}/{payload.get('source_format')}]",
        f"[Раздел: {payload.get('section')}]",
    ]
    if payload.get("program_code") or payload.get("program_name"):
        parts.append(f"[Программа: {payload.get('program_code') or ''} {payload.get('program_name') or ''}]")
    if payload.get("section_type"):
        parts.append(f"[Секция: {payload.get('section_type')}]")
    return "".join(parts)


def _chunk_text_by_sentences(
    content: str,
    source_name: str,
    section_name: str,
    *,
    source_format: str,
    source_group: str,
    doc_type: str,
    metadata: dict | None = None,
    start_index: int = 0,
) -> list:
    """Split text content into sentence-boundary chunks with overlap."""
    sentences = [sentence.strip() for sentence in _SENT_SPLIT.split(content) if sentence.strip()]
    chunks = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= MAX_CHARS_PER_CHUNK:
            current = current + " " + sentence
        else:
            if current.strip():
                payload = _base_payload(
                    source_name=source_name,
                    section_name=section_name,
                    source_format=source_format,
                    source_group=source_group,
                    doc_type=doc_type,
                    metadata=metadata,
                    chunk_index=start_index + len(chunks),
                    content_type="text_part",
                )
                payload["text"] = current
                chunks.append((f"{_embedding_prefix(payload)} {current}", payload))
            overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current = (overlap_text + " " + sentence).strip()

    if current.strip():
        payload = _base_payload(
            source_name=source_name,
            section_name=section_name,
            source_format=source_format,
            source_group=source_group,
            doc_type=doc_type,
            metadata=metadata,
            chunk_index=start_index + len(chunks),
            content_type="text_part",
        )
        payload["text"] = current
        chunks.append((f"{_embedding_prefix(payload)} {current}", payload))

    return chunks


def build_chunks(
    raw_text: str,
    source_name: str,
    *,
    metadata: dict | None = None,
    source_format: str = "md",
    source_group: str = "curated",
    doc_type: str = "handbook",
) -> list:
    sections = extract_sections_markdown(raw_text)
    chunks = []

    for section_name, content in sections.items():
        if not content:
            continue

        if _is_markdown_table(content):
            if _is_noisy_table(content):
                logger.debug("Skipping noisy table in section '%s'", section_name)
                continue
            payload = _base_payload(
                source_name=source_name,
                section_name=section_name,
                source_format=source_format,
                source_group=source_group,
                doc_type=doc_type,
                metadata=metadata,
                chunk_index=len(chunks),
                content_type="table",
            )
            payload["text"] = content
            chunks.append((f"{_embedding_prefix(payload)} {content}", payload))
        else:
            chunks.extend(
                _chunk_text_by_sentences(
                    content,
                    source_name,
                    section_name,
                    source_format=source_format,
                    source_group=source_group,
                    doc_type=doc_type,
                    metadata=metadata,
                    start_index=len(chunks),
                )
            )

    return chunks


def build_markdown_chunks(markdown_text: str, source_name: str, metadata: dict | None = None) -> list:
    body, frontmatter = _strip_frontmatter(markdown_text)
    inferred = get_markdown_metadata(source_name)
    merged = {**inferred, **frontmatter, **(metadata or {})}
    return build_chunks(
        body,
        source_name,
        metadata=merged,
        source_format="md",
        source_group=merged.get("source_group", "curated"),
        doc_type=merged.get("doc_type", "program_profile"),
    )
