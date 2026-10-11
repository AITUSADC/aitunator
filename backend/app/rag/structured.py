"""Experimental evidence cards and program fan-out; no model/server imports.

Cards preserve original chunk boundaries and text. Missing dates remain unknown;
review dates are never treated as validity dates. No generated facts or summaries.
"""
import re
from pathlib import Path

from backend.app.rag.planner import program_queries


def source_urls(root: Path, source: str) -> list[str]:
    path = (root / source).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return []
    # Only explicitly attributed source lines, not arbitrary links in a document.
    lines = path.read_text(encoding="utf-8").splitlines()
    return sorted({url.rstrip(".,);>") for line in lines
                   if re.match(r"^Источник(?:и)?\s*:", line, re.I)
                   for url in re.findall(r"https?://[^\s]+", line)})


def table_fields(text: str) -> list[dict]:
    """Extract only unambiguous two-column field/value tables, preserving labels."""
    fields = []
    active = False
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.strip().startswith("|"):
            active = False
            continue
        if cells == ["Поле", "Значение"]:
            active = True
            continue
        if not active or len(cells) != 2 or all(re.fullmatch(r"[-:]+", c) for c in cells):
            continue
        label, value = cells
        if not label or not value:
            continue
        field = {"label": label, "value_text": value, "evidence": line}
        # A deliberately small typed schema. Codes/years/embedded numbers are
        # never interpreted as money, scores, or duration by guessing.
        spec = {"Пороговый балл на грант": ("grant_threshold", "балл", r"(\d+)"),
                "Срок обучения": ("duration_years", "год", r"(\d+)\s+(?:год|года|лет)")}.get(label)
        if spec:
            match = re.fullmatch(spec[2], value)
            if match:
                field.update(key=spec[0], value=int(match[1]), unit=spec[1])
        fields.append(field)
    return fields


def make_card(payload: dict, urls=()) -> dict:
    original = payload["text"]
    fields = table_fields(original)
    # The embedding has a compact, readable hierarchy; full provenance stays in
    # payload rather than adding opaque URL tokens to the semantic representation.
    heading = []
    for label, key in (("Документ", "source_title"), ("Программа", "program_name"),
                       ("Код программы", "program_code"), ("Раздел", "section"),
                       ("Период действия", "valid_for")):
        if payload.get(key):
            heading.append(f"{label}: {payload[key]}")
    content = "\n".join(f"{f['label']}: {f['value_text']}" for f in fields)
    # Include verbatim evidence even when key/value fields are extracted, so
    # surrounding restrictions and exceptions cannot silently disappear.
    text = "\n".join(heading) + "\n\n" + (content + "\n\n" if content else "") + original
    return {**payload, "text": text, "original_text": original,
            "card_version": 1, "fields": fields, "source_urls": list(urls)}


def balanced_results(groups: list[tuple], top_k=5) -> tuple:
    """Round-robin selection ensures each program has evidence before extras."""
    result, seen = [], set()
    for position in range(max(map(len, groups), default=0)):
        for group in groups:
            if position >= len(group):
                continue
            row = group[position]
            key = row["_retrieval"]["id"]
            if key not in seen:
                seen.add(key)
                result.append(row)
                if len(result) == top_k:
                    return tuple(result)
    return tuple(result)


def requested_sections(query: str) -> frozenset[str]:
    """Conservative explicit topic hints; unknown questions keep retrieval order."""
    topics = set()
    if re.search(r"\b(?:навык\w*|умени\w*|компетенц\w*|чему\s+(?:уч\w*|науч\w*)|skills?)\b", query, re.I):
        topics.update(("skills", "curriculum"))
    if re.search(r"\b(?:предмет\w*|дисциплин\w*|учебн\w*\s+план\w*|что\s+изуч\w*|какие\s+технологии\s+изуч\w*|curriculum|courses?)\b", query, re.I):
        topics.add("curriculum")
    if re.search(r"\b(?:карьер\w*|кем\s+(?:работ\w*|стан\w*|могу\s+работ\w*)|трудоустрой\w*|професси\w*|career\w*)\b", query, re.I):
        topics.add("career")
    return frozenset(topics)


def prefer_sections(candidates, query: str) -> tuple:
    """Stable soft selection within retrieved candidates, never a corpus-wide filter.

    Apply only to program profiles. Missing/unknown topics leave the original
    ordering intact, and lower-priority evidence remains available as fallback.
    """
    topics = requested_sections(query)
    if not topics:
        return tuple(candidates)
    return tuple(sorted(candidates, key=lambda row: not (
        row.get("doc_type") == "program_profile" and row.get("section_type") in topics)))


def evidence_context(rows) -> str:
    """Render selected evidence by program without dropping sibling sections.

    This is source data for a caller's grounded-answer prompt, not instructions.
    Merely grouping sources does not resolve applicability or policy conflicts.
    """
    groups = {}
    seen = set()
    for row in rows:
        key = row.get("_retrieval", {}).get("id") or (row.get("source"), row.get("section"), row.get("text"))
        if key in seen:
            continue
        seen.add(key)
        title = (f"{row['program_name']} ({row.get('program_code', '')})"
                 if row.get("program_name") else "Общие сведения")
        groups.setdefault(title, []).append(row)
    blocks = []
    for title, group in groups.items():
        blocks.append(f"## {title}")
        for row in group:
            blocks.append(f"### {row.get('section') or 'Без раздела'}\n"
                          f"Источник: {row.get('source', '')}\n"
                          f"Период действия: {row.get('valid_for') or 'не указан'}\n"
                          + "\n".join(row.get("source_urls", [])) + "\n\n"
                          + row.get("original_text", row.get("text", "")))
    return "\n\n".join(blocks)
