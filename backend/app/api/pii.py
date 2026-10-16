"""Personal data (PII) detection for questions, assistant answers, and feedback comments.

Regex filters catch known structured formats (Kazakhstan 12-digit IIN, phone numbers,
emails, payment cards, identity document numbers, and explicit PII self-disclosures).
Per docs/contract.md, any turn containing PII in the question or answer is not saved
to SQLite (message_id=null), and feedback comments containing PII are rejected.
"""
from __future__ import annotations

import re

# 12-digit Kazakhstan IIN (either standalone 12 digits or separated by spaces/hyphens after an IIN keyword)
_IIN_RE = re.compile(r"(?<!\d)\d{12}(?!\d)|(?:\b(?:иин|iin|жсн)\b[\s:№#-]*\d{4}[\s-]?\d{4}[\s-]?\d{4})", re.I)

# Kazakhstan / international phone numbers: +7 (7xx) xxx-xx-xx, 87xxxxxxxxx, +77xxxxxxxxx
_PHONE_RE = re.compile(
    r"(?:(?:\+7|(?<!\d)8)[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2})\b"
    r"|(?:\+\d{1,3}[\s\-()]*\d{2,4}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2})\b"
)

# Standard email addresses (excluding official @astanait.edu.kz institutional contacts in answers)
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
_ALLOWED_EMAIL_DOMAINS = frozenset({"astanait.edu.kz"})

# 16-digit bank card numbers (4 groups of 4 digits or 16 contiguous digits)
_BANK_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[\s\-]){3}\d{4}(?!\d)|(?<!\d)\d{16}(?!\d)")

# Identity card / passport numbers accompanied by document keywords
_DOC_NUMBER_RE = re.compile(
    r"\b(?:удостоверени\w*|паспорт\w*|id\s*card|номер\s+документа|серия\s+и\s+номер)\b[^\n.]{0,30}?\b[A-ZА-Я№]?\d{7,9}\b",
    re.I,
)


def contains_pii(text: str | None, *, allow_official_contacts: bool = False) -> bool:
    """Return True if text contains recognizable personal data patterns."""
    if not text or not text.strip():
        return False

    if _IIN_RE.search(text):
        return True

    if _BANK_CARD_RE.search(text):
        return True

    if _DOC_NUMBER_RE.search(text):
        return True

    for match in _EMAIL_RE.finditer(text):
        domain = match.group(1).lower().rstrip(".")
        if allow_official_contacts and domain in _ALLOWED_EMAIL_DOMAINS:
            continue
        return True

    if _PHONE_RE.search(text):
        if allow_official_contacts:
            mobile_re = re.compile(
                r"(?:\+7|(?<!\d)8)[\s\-()]*7(?:0[0-9]|47|7[0-9])[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b"
            )
            if mobile_re.search(text):
                return True
        else:
            return True

    return False
