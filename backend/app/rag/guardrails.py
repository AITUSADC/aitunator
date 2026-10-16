"""Conservative deterministic checks before sending questions to an LLM.

These checks cover common structured formats, not all personal information.
API persistence filtering remains a separate responsibility.
"""
import re

CONTACT = "Обратитесь в приёмную комиссию Astana IT University."
REFUSAL = "Не могу обрабатывать личные данные или давать гарантии поступления и получения гранта. " + CONTACT
NO_DATA = "В доступных материалах недостаточно информации для надёжного ответа. " + CONTACT

_PII = re.compile(
    r"(?<!\d)\d{12}(?!\d)|\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"
    r"|(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)"
    r"|(?<!\w)(?:\+\d|8)[\d ()-]{8,}\d"
    r"|\b(?:иин|iin|жсн|паспорт\w*|удостоверени\w*)\s*[:№#-]?\s*[A-ZА-Я]?\d[\d -]{6,}", re.I)
_PRIVATE = re.compile(
    r"личн\w*\s+(?:номер|телефон|контакт|данн)|домашн\w*\s+адрес"
    r"|телефон\s+(?:ректора|декана|преподавателя|студента)"
    r"|чуж\w*\s+(?:иин|данн)|(?:private|personal)\s+(?:phone|contact|data|address)", re.I)
_GUARANTEE = re.compile(
    r"(?:гарантир\w*|точно|100\s*%|обеща\w*)[^.!?]{0,60}(?:поступ|грант)"
    r"|(?:поступ|грант)[^.!?]{0,60}(?:гарантир|точно|100\s*%)"
    r"|guarantee\w*[^.!?]{0,60}(?:admission|scholarship)", re.I)


def has_personal_data(text):
    return bool(_PII.search(text))


def blocked(text):
    return has_personal_data(text) or bool(_PRIVATE.search(text) or _GUARANTEE.search(text))
