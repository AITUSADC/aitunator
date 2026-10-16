"""Personal data (PII) detection for questions, assistant answers, and feedback comments.

Per project security requirements:
- Regex filters detect Kazakhstan 12-digit IIN, bank card numbers, passport/ID numbers,
  phone numbers, and email addresses.
- In assistant answers, ONLY specific, verified public institutional contacts of Astana IT University
  are permitted (e.g. info@astanait.edu.kz, admission@astanait.edu.kz, reception +7 7172 645710).
  Arbitrary personal email addresses (even with @astanait.edu.kz domain) and personal/international
  phone numbers (+1..., +7 701..., etc.) are strictly treated as personal data.
- Turns containing PII in question or answer are never saved to SQLite (message_id=null).
- Feedback comments containing PII are rejected with 422 pii_detected.
"""
from __future__ import annotations

import re

# 12-digit Kazakhstan IIN
_IIN_RE = re.compile(
    r"(?<!\d)\d{12}(?!\d)|(?:\b(?:иин|iin|жсн)\b[\s:№#-]*\d{4}[\s-]?\d{4}[\s-]?\d{4})",
    re.I,
)

# 16-digit payment card numbers
_BANK_CARD_RE = re.compile(
    r"(?<!\d)(?:\d{4}[\s\-]){3}\d{4}(?!\d)|(?<!\d)\d{16}(?!\d)"
)

# Identity document / passport numbers
_DOC_NUMBER_RE = re.compile(
    r"\b(?:удостоверени\w*|паспорт\w*|id\s*card|номер\s+документа|серия\s+и\s+номер)\b[^\n.]{0,30}?\b[A-ZА-Я№]?\d{7,9}\b",
    re.I,
)

# Any email pattern
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")

# Strictly allowed official public institutional email addresses for AITU
_OFFICIAL_PUBLIC_EMAILS = frozenset({
    "info@astanait.edu.kz",
    "admission@astanait.edu.kz",
    "reception@astanait.edu.kz",
    "press@astanait.edu.kz",
    "support@astanait.edu.kz",
    "career@astanait.edu.kz",
    "mobility@astanait.edu.kz",
})

# Any phone pattern (Kazakhstan, Russian, international like +1 202 555 0123)
_ANY_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s\-()]*)?(?:\(?\d{2,4}\)?[\s\-]*)?\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b"
    r"|(?:\+\d{1,3}[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{4})\b"
    r"|(?:\b8[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b)"
)

# Strictly allowed official university landline reception numbers (Astana city code 7172)
_OFFICIAL_LANDLINE_DIGITS = frozenset({
    "77172645710",  # +7 (7172) 645-710 (General reception)
    "87172645710",
    "77172645707",  # +7 (7172) 645-707 (Admissions call-center)
    "87172645707",
    "77172645700",  # +7 (7172) 645-700
    "87172645700",
})


def _normalize_phone_digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def contains_pii(text: str | None, *, allow_official_contacts: bool = False) -> bool:
    """Return True if text contains recognizable personal data."""
    if not text or not text.strip():
        return False

    if _IIN_RE.search(text):
        return True

    if _BANK_CARD_RE.search(text):
        return True

    if _DOC_NUMBER_RE.search(text):
        return True

    # Check email addresses
    for match in _EMAIL_RE.finditer(text):
        email_str = match.group(0).lower().rstrip(".")
        if allow_official_contacts and email_str in _OFFICIAL_PUBLIC_EMAILS:
            continue
        return True

    # Check phone numbers
    for match in _ANY_PHONE_RE.finditer(text):
        phone_match = match.group(0)
        digits = _normalize_phone_digits(phone_match)
        if len(digits) < 7:
            continue
        if allow_official_contacts:
            if digits in _OFFICIAL_LANDLINE_DIGITS or (len(digits) >= 10 and digits[-10:] in {"7172645710", "7172645707", "7172645700"}):
                continue
        return True

    return False
