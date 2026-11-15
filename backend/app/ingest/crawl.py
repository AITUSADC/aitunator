"""Build data/md from astanait.edu.kz: fetch pages (robots.txt, <=1 rps, raw cache) and convert to Markdown."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import hashlib
from itertools import repeat
import os
from pathlib import Path
import re
import time
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import httpx
from markdownify import markdownify

from backend.app.config import KNOWLEDGE_BASE_DIR
from backend.app.ingest.documents import RAW_DOCS_DIR, cached_markdown, document_links, fetch_documents, drop_signature_pages, strip_personal_tables

OCR_WORKERS = min(8, os.cpu_count() or 1)  # ponytail: fixed for ~1 GB RAM per worker on large scans

SITE = "https://astanait.edu.kz"
USER_AGENT = "aitunator-kb-builder"
RAW_DIR = "runtime/raw_html"

# Target file -> (title, site pages). File names must match backend.app.rag.programs registries.
SOURCES: dict[str, tuple[str, list[str]]] = {
    "programs/ai_business.md": ("AI Business", ["/ru/aibusiness-bachelor"]),
    "programs/big_data_analysis.md": ("Big Data Analysis", ["/ru/BigDataAnalysis-bachelor"]),
    "programs/it.md": ("Computer Science", ["/ru/Computer-Science-bachelor"]),
    "programs/cs.md": ("Cybersecurity", ["/ru/Cybersecurity"]),
    "programs/digital_journalism.md": ("Digital Journalism", ["/ru/dj"]),
    "programs/digital_public_administration.md": ("Digital Public Administration", ["/ru/digital-public-administration-bachelor"]),
    "programs/digital_technologies_nuclear_power.md": ("Digital Technologies in Nuclear Power Engineering", ["/ru/digitaltechnologiesin-nuclear-bachelor"]),
    "programs/electronic_engineering.md": ("Electronic Engineering", ["/ru/electronic-engineering-bachelor"]),
    "programs/industrial_internet_of_things.md": ("Industrial Internet of Things", ["/ru/industrial-internet-things-bachelor"]),
    "programs/it_entrepreneurship.md": ("IT Entrepreneurship", ["/ru/itentrepre-bachelor"]),
    "programs/it_management.md": ("IT Management", ["/ru/itmanagement-bachelor"]),
    "programs/mathematical_computational_science.md": ("Mathematical and Computational Science", ["/ru/mathematicalcomputational-science-bachelor"]),
    "programs/media_technologies.md": ("Media Technologies", ["/ru/mediatechbachelor"]),
    "programs/smart_security_technologies.md": ("Smart Security Technologies", ["/ru/smartsecuritytechnologies-bachelor"]),
    "programs/smart_technologies.md": ("Smart Technologies", ["/ru/smarttechnologies-bachelor"]),
    "programs/software_engineering.md": ("Software Engineering", ["/ru/Software-Engineering-bachelor"]),
    "admissions.md": ("Поступление в AITU", ["/ru/bachelor", "/ru/how-to-apply", "/ru/admission-after-college",
                                             "/ru/aitu-excellence-test", "/ru/creative-exams"]),
    "student_grants.md": ("Гранты и стипендии AITU", ["/ru/educational-grants", "/ru/student-grants"]),
    "academic_calendar.md": ("Академический календарь AITU", ["/ru/academic-calendar"]),
    "academic_rules.md": ("Академические правила AITU", ["/ru/academic-opportunities"]),
    "academic_mobility.md": ("Международная и академическая мобильность AITU", [
        "/ru/outg-academ-mobility", "/ru/inc-academ-mobility", "/ru/inter-credit-mobility", "/ru/double-degree-program"]),
    "career_guide.md": ("Центр карьеры AITU", ["/ru/career-center"]),
    "contacts.md": ("Контакты AITU", ["/ru/contacts"]),
    "first_year_guide.md": ("Первокурснику AITU", ["/ru/study-at-aitu", "/ru/additional-materials-for-students"]),
    "library.md": ("Научная библиотека AITU", ["/ru/scientific-library", "/ru/about-us-page-library"]),
    "military_department.md": ("Военная кафедра AITU", ["/ru/military-department"]),
    "student_clubs.md": ("Студенческие клубы AITU", ["/ru/student-clubs", "/ru/aitusa"]),
    "student_housing.md": ("Общежитие AITU", ["/ru/students-dormitory"]),
    "student_life.md": ("Студенческая жизнь AITU", ["/ru/student-life-page", "/ru/sports-AITU"]),
    "student_support.md": ("Поддержка студентов AITU", ["/ru/psychological-counseling-center", "/ru/medical-services"]),
    "university_profile.md": ("Об Astana IT University", ["/ru/about-aitu", "/ru/licenses-accreditations"]),
    "tuition_and_discounts.md": ("Стоимость обучения и скидки AITU", []),  # filled from linked documents
}

_CHROME_TAGS = ["script", "style", "header", "nav", "footer", "iframe", "svg", "img", "button", "noscript", "form"]
# Lines with personal data are dropped: a name with a patronymic, a 12-digit IIN,
# a non-role mailbox (staff mail is often surname@) or a mobile number (landlines are +7 7172).
_PERSONAL_LINE = re.compile(
    r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+\s+[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+(?:вич|вна|улы|ұлы|кызы|қызы)\b"
    r"|[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{2,}\s+[А-ЯЁӘҒҚҢӨҰҮҺІ]\.\s?[А-ЯЁӘҒҚҢӨҰҮҺІ]\."
    r"|[А-ЯЁӘҒҚҢӨҰҮҺІ]\.\s?[А-ЯЁӘҒҚҢӨҰҮҺІ]\.\s?[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{2,}"
    r"|(?<!\d)\d{12}(?!\d)"
)
_UP, _LOW = "А-ЯЁӘҒҚҢӨҰҮҺІ", "а-яёәғқңөұүһі"
# A line holding only "Surname Name" (signatures, student government) without a patronymic.
_NAME_ONLY_LINE = re.compile(
    rf"^\W{{0,4}}[{_UP}][{_LOW}]+(?:ов|ев|ёв|ин|ын|ова|ева|ёва|ина|ына|бек|бекова|ұлы|улы|қызы|кызы)"
    rf"\s+[{_UP}][{_LOW}]{{2,}}(?:\s+[{_UP}][{_LOW}]{{2,}})?.{{0,4}}$")
_EMAIL = re.compile(r"([\w.+-]+)@\w")  # OCR turns dots into commas, so the domain is not checked
_ROLE_MAILBOX = re.compile(
    r"(?:info|admission|reception|press|pr|support|career|mobility|library|international|\w+_exams)", re.I)
_PHONE = re.compile(r"(?:\+\s*7|\b8)[\d\s()-]{9,}\d")
_HEADING = re.compile(r"^#{1,6}\s")


def _is_personal(line: str) -> bool:
    if _PERSONAL_LINE.search(line) or _NAME_ONLY_LINE.match(line):
        return True
    if any(not _ROLE_MAILBOX.fullmatch(m.group(1)) for m in _EMAIL.finditer(line)):
        return True
    for match in _PHONE.finditer(line):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) == 11 and digits[1] == "7" and digits[1:5] != "7172":
            return True
    return False


def _clean_markdown(md: str) -> str:
    lines = []
    for line in md.splitlines():
        line = line.rstrip()
        if _is_personal(line):
            continue
        # Program pages show "Структура программы" only in the sidebar; the discipline
        # tables start with the "**ООД**" legend and would otherwise merge into the previous section.
        if line.startswith("**ООД**") and "## Структура программы" not in lines:
            lines += ["## Структура программы", ""]
        lines.append("## " + line.lstrip("# ") if line.startswith("# ") else line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    # Drop headings with no content before the next heading of the same or higher level.
    parts = re.split(r"(?m)^(?=#{1,6}\s)", text)
    kept = []
    for i, part in enumerate(parts):
        heading, _, body = part.partition("\n")
        if _HEADING.match(heading) and not body.strip():
            level = len(heading) - len(heading.lstrip("#"))
            following = parts[i + 1] if i + 1 < len(parts) else ""
            next_level = len(following) - len(following.lstrip("#")) if following else 0
            if not following or next_level <= level:
                continue
        kept.append(part.strip())
    return "\n\n".join(p for p in kept if p)


def html_to_blocks(html: str) -> list[str]:
    """Return non-empty Markdown blocks, one per page section (Next.js `section-*` containers)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_CHROME_TAGS):
        tag.decompose()
    containers = soup.select('div[id^="section-"]') or [soup.body or soup]
    blocks = [_clean_markdown(markdownify(str(c), heading_style="ATX", strip=["a"])) for c in containers]
    return [b for b in blocks if b]


def site_chrome(pages: list[list[str]], share: float = 0.5) -> set[str]:
    """Blocks present on more than `share` of all pages are site-wide chrome (footers, contact widgets)."""
    counts = Counter(block for blocks in pages for block in set(blocks))
    return {block for block, n in counts.items() if n > share * len(pages)}


def build_document(title: str, pages: list[tuple[str, list[str]]], chrome: set[str], updated_at: str) -> str:
    body = [block for _, blocks in pages for block in blocks if block not in chrome]
    text = "\n\n".join(body)
    if not text.startswith("## "):
        text = f"## {title}\n\n{text}"
    urls = ", ".join(url for url, _ in pages)
    return f"---\ntitle: {title}\nsource_url: [{urls}]\nlanguage: ru\nupdated_at: {updated_at}\n---\n\n{text}\n"


def fetch_pages(paths: list[str], raw_dir: Path, client: httpx.Client, delay: float = 1.0) -> dict[str, str]:
    """Fetch allowed paths into raw_dir (cached by path) and return {path: html}."""
    robots = RobotFileParser()
    robots.parse(client.get("/robots.txt").text.splitlines())
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages = {}
    for path in paths:
        if not robots.can_fetch(USER_AGENT, path):
            continue
        cached = raw_dir / (hashlib.md5(path.encode()).hexdigest() + ".html")
        if not cached.exists():
            time.sleep(delay)
            response = client.get(path)
            if response.status_code == 404:
                print(f"skip {path}: 404")
                continue
            response.raise_for_status()
            cached.write_text(response.text, encoding="utf-8")
        pages[path] = cached.read_text(encoding="utf-8")
    return pages


def _single_threaded_worker() -> None:
    # pymupdf layout creates ONNX sessions with one thread per core and exposes no option;
    # with several workers that oversubscribes the CPU (load ~10x cores), so force one thread.
    import onnxruntime as ort

    class SessionOptions(ort.SessionOptions):
        def __init__(self):
            super().__init__()
            self.intra_op_num_threads = self.inter_op_num_threads = 1

    ort.SessionOptions = SessionOptions


def _raw_markdown(file_id: str, data: bytes, raw_dir: Path) -> str | None:
    try:
        return cached_markdown(file_id, data, raw_dir)
    except RuntimeError as exc:  # OCR needs tesseract; the file is retried on the next run
        print(f"skip document {file_id}: {exc}")
        return None


def _linked_documents(pages: dict[str, str], raw_dir: Path) -> dict[str, list[tuple[str, list[str]]]]:
    """Download Drive/Docs files linked from pages and group their Markdown by target file."""
    page_links = [document_links(pages[p], target, title)
                  for target, (title, urls) in SOURCES.items() for p in urls if p in pages]
    common = site_chrome([[file_id for file_id, _, _ in links] for links in page_links])
    wanted = {}
    for links in page_links:
        for file_id, title, target in links:
            if file_id not in common:
                wanted.setdefault(file_id, (title, target))
    # Google robots.txt disallows /uc; downloading anyway was an explicit project decision (1 rps).
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60) as client:
        files = fetch_documents(list(wanted), raw_dir, client)

    # OCR is CPU-bound and single-threaded per page: convert documents in parallel, largest first.
    for var in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    order = sorted(files, key=lambda file_id: -len(files[file_id]))
    with ProcessPoolExecutor(max_workers=OCR_WORKERS, initializer=_single_threaded_worker) as pool:
        raw = dict(zip(order, pool.map(_raw_markdown, order, [files[i] for i in order], repeat(raw_dir))))

    grouped: dict[str, list[tuple[str, list[str]]]] = {}
    for file_id, text in raw.items():
        title, target = wanted[file_id]
        if text is None:
            continue
        md = _clean_markdown(strip_personal_tables(drop_signature_pages(text)))
        if len(md) < 200:
            print(f"skip document {file_id} ({title}): no text")
            continue
        grouped.setdefault(target, []).append(
            (f"https://drive.google.com/file/d/{file_id}/view", [f"## {title}\n\n{md}"]))
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Crawl astanait.edu.kz into data/md")
    parser.add_argument("--out", type=Path, default=Path(KNOWLEDGE_BASE_DIR))
    parser.add_argument("--raw", type=Path, default=Path(RAW_DIR))
    parser.add_argument("--raw-docs", type=Path, default=Path(RAW_DOCS_DIR))
    args = parser.parse_args()

    paths = sorted({p for _, urls in SOURCES.values() for p in urls})
    with httpx.Client(base_url=SITE, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30) as client:
        pages = fetch_pages(paths, args.raw, client)

    blocks = {path: html_to_blocks(html) for path, html in pages.items()}
    chrome = site_chrome(list(blocks.values()))
    documents = _linked_documents(pages, args.raw_docs)
    today = date.today().isoformat()
    for target, (title, urls) in SOURCES.items():
        doc_pages = [(SITE + p, blocks[p]) for p in urls if blocks.get(p)] + documents.get(target, [])
        if not doc_pages:
            print(f"{target}: no content, skipped")
            continue
        out = args.out / target
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_document(title, doc_pages, chrome, today), encoding="utf-8")
        print(f"{target}: {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
