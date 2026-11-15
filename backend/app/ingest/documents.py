"""Documents linked from site pages (Google Drive / Docs): link selection, download and PDF/DOCX -> Markdown."""
from pathlib import Path
import re
import subprocess
import time

from bs4 import BeautifulSoup
import httpx
from markdownify import markdownify

RAW_DOCS_DIR = "runtime/raw_docs"
DRIVE_DOWNLOAD = "https://drive.google.com/uc"

_DOC_ID = re.compile(
    r"(?:drive\.google\.com/(?:file/d/|open\?id=)|docs\.google\.com/(?:document|spreadsheets|presentation)/d/)([\w-]+)")
# Personal lists (grant holders, dorm residents, exam results), documents with no answer value and
# curriculum scans (the same tables are already on program pages as HTML; their OCR is mostly noise).
_SKIP_TITLE = re.compile(
    r"[сc]писок|результат|отзыв|сертификат|лицензи|аудитор|финансов|договор|образ[еe]ц|заявлени"
    r"|программа развития|план развития|учебный план", re.I)
_ROUTES = [
    (re.compile(r"прейскурант|скидк", re.I), "tuition_and_discounts.md"),
    (re.compile(r"академическая политика", re.I), "academic_rules.md"),
    (re.compile(r"академический календарь", re.I), "academic_calendar.md"),
]


def document_links(html: str, page_target: str, page_title: str) -> list[tuple[str, str, str]]:
    """Return unique (file_id, title, target_file) for useful Drive/Docs links on a page."""
    links, seen = [], set()
    for anchor in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        match = _DOC_ID.search(anchor["href"])
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not match or match.group(1) in seen or _SKIP_TITLE.search(title):
            continue
        seen.add(match.group(1))
        target = next((t for pattern, t in _ROUTES if pattern.search(title)), page_target)
        links.append((match.group(1), title if len(title) >= 12 else f"Документ: {page_title}", target))
    return links


_NAME_COLUMN = re.compile(r"фамилия|\bф\.?\s?и\.?\s?о\b", re.I)


def strip_personal_tables(md: str) -> str:
    """Drop tables whose header has a name column (people lists, form templates, staff contacts)."""
    kept, drop = [], False
    for line in md.splitlines():
        if not line.startswith("|"):
            drop = False
        elif not kept or not kept[-1].startswith("|") or drop:
            drop = drop or bool(_NAME_COLUMN.search(line))
        if not drop:
            kept.append(line)
    return "\n".join(kept)


_SIGNATURE = re.compile(r"\(\s*подпись|фио\s*/|/\s*и?фио|лист согласования|signature", re.I)


def drop_signature_pages(md: str) -> str:
    """Approval sheets name students and officials (OCR splits names across lines); they carry no answers."""
    pages = md.split("\f")
    if len(pages) == 1:  # DOCX: clean text, names are caught line by line
        return md
    return "\n\n".join(page for page in pages if not _SIGNATURE.search(page))


def fetch_documents(ids: list[str], raw_dir: Path, client: httpx.Client, delay: float = 1.0) -> dict[str, bytes]:
    """Download files into raw_dir (cached by id); login pages and errors are skipped, not cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for file_id in ids:
        cached = raw_dir / f"{file_id}.bin"
        if not cached.exists():
            time.sleep(delay)
            response = client.get(DRIVE_DOWNLOAD, params={"export": "download", "id": file_id})
            if response.status_code != 200 or response.content.lstrip()[:1] == b"<":
                print(f"skip document {file_id}: not downloadable")
                continue
            cached.write_bytes(response.content)
        files[file_id] = cached.read_bytes()
    return files


def _readable(text: str) -> bool:
    words = re.findall(r"\w+", text)
    real = [w for w in words if len(w) >= 3 and w.isalpha()]
    return len(words) >= 5 and len(real) >= 0.5 * len(words)


def _clean_ocr_text(text: str) -> str:
    """Keep lines made mostly of real words (3+ letters) or numbers; OCR of table grids is mostly noise."""
    kept = []
    for line in text.splitlines():
        tokens = re.findall(r"\w+", line)
        ok = [t for t in tokens if (t.isalpha() and len(t) >= 3) or t.isdigit()]
        if tokens and any(t.isalpha() for t in ok) and len(ok) >= 0.6 * len(tokens):
            kept.append(line.strip())
    return "\n".join(kept)


def _osd_rotation(page) -> int:
    """Clockwise rotation suggested by tesseract OSD (landscape tables scanned on portrait pages)."""
    png = page.get_pixmap(dpi=150).tobytes("png")
    result = subprocess.run(["tesseract", "stdin", "-", "--psm", "0"], input=png, capture_output=True)
    match = re.search(rb"Rotate: (\d+)", result.stdout)
    return int(match.group(1)) if match else 0


def _ocr_page(page) -> str:
    page.set_rotation((page.rotation + _osd_rotation(page)) % 360)
    textpage = page.get_textpage_ocr(language="rus+eng", dpi=300, full=True)
    return _clean_ocr_text(page.get_text(textpage=textpage))


def _pdf_to_markdown(data: bytes) -> str:
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(stream=data, filetype="pdf")
    readable = [i for i, page in enumerate(doc) if _readable(page.get_text())]
    chunks = pymupdf4llm.to_markdown(doc, pages=readable, page_chunks=True, use_ocr=False, show_progress=False) \
        if readable else []
    markdown = {chunk["metadata"]["page_number"] - 1: chunk["text"] for chunk in chunks}
    # Scans and broken text layers get plain OCR text; their table structure is lost.
    return "\f".join(markdown[i] if i in markdown else _ocr_page(page) for i, page in enumerate(doc))


def _docx_to_markdown(data: bytes) -> str:
    from io import BytesIO
    import mammoth

    try:
        html = mammoth.convert_to_html(BytesIO(data)).value
    except Exception:  # xlsx/pptx are zip archives too; they are not supported
        return ""
    return markdownify(html, heading_style="ATX", strip=["a", "img"])


def cached_markdown(file_id: str, data: bytes, raw_dir: Path) -> str:
    """Raw (uncleaned) Markdown is cached so PII rule changes do not require another OCR pass."""
    cached = raw_dir / f"{file_id}.md"
    if not cached.exists():
        cached.write_text(document_to_markdown(data), encoding="utf-8")
    return cached.read_text(encoding="utf-8")


def document_to_markdown(data: bytes) -> str:
    if data.startswith(b"%PDF"):
        return _pdf_to_markdown(data)
    if data.startswith(b"PK"):
        return _docx_to_markdown(data)
    return ""
