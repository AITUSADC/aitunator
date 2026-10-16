"""Personal data (PII) detection for questions, assistant answers, and feedback comments."""
from __future__ import annotations

import re

_IIN_RE = re.compile(
    r"(?<!\d)\d{12}(?!\d)|(?:\b(?:иин|iin|жсн)\b[\s:№#-]*\d{4}[\s-]?\d{4}[\s-]?\d{4})",
    re.I,
)
_BANK_CARD_RE = re.compile(
    r"(?<!\d)(?:\d{4}[\s\-]){3}\d{4}(?!\d)|(?<!\d)\d{16}(?!\d)"
)
_DOC_NUMBER_RE = re.compile(
    r"\b(?:удостоверени\w*|паспорт\w*|id\s*card|номер\s+документа|серия\s+и\s+номер)\b[^\n.]{0,30}?\b[A-ZА-Я№]?\d{7,9}\b",
    re.I,
)
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")

_OFFICIAL_PUBLIC_EMAILS = frozenset({
    "info@astanait.edu.kz",
    "admission@astanait.edu.kz",
    "reception@astanait.edu.kz",
    "press@astanait.edu.kz",
    "support@astanait.edu.kz",
    "career@astanait.edu.kz",
    "mobility@astanait.edu.kz",
})

_ANY_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s\-()]*)?(?:\(?\d{2,4}\)?[\s\-]*)?\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b"
    r"|(?:\+\d{1,3}[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{4})\b"
    r"|(?:\b8[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b)"
)

_OFFICIAL_FULL_PHONES = frozenset({
    "77172645710",
    "87172645710",
    "77172645707",
    "87172645707",
    "77172645700",
    "87172645700",
})


def contains_pii(text: str | None, *, allow_official_contacts: bool = False) -> bool:
    if not text or not text.strip():
        return False

    if _IIN_RE.search(text):
        return True
    if _BANK_CARD_RE.search(text):
        return True
    if _DOC_NUMBER_RE.search(text):
        return True

    for match in _EMAIL_RE.finditer(text):
        email_str = match.group(0).lower().rstrip(".")
        if allow_official_contacts and email_str in _OFFICIAL_PUBLIC_EMAILS:
            continue
        return True

    for match in _ANY_PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 7:
            continue
        if allow_official_contacts and digits in _OFFICIAL_FULL_PHONES:
            continue
        return True

    return False
