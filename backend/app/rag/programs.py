from __future__ import annotations

from pathlib import Path
import re


PROGRAM_SITE_REVIEWED_AT = "2026-06-29"


PROGRAM_REGISTRY: dict[str, dict] = {
    "AIB_DP_.pdf": {
        "program_code": "6B04103",
        "program_slug": "aib",
        "program_name": "AI Business",
        "school": "School of Creative Industries",
    },
    "BDA_DP_.pdf": {
        "program_code": "6B06103",
        "program_slug": "bda",
        "program_name": "Big Data Analysis",
        "school": "School of Artificial Intelligence and Data Science",
    },
    "CS_DP_.pdf": {
        "program_code": "6B06301",
        "program_slug": "cs",
        "program_name": "Cyber Security",
        "school": "School of Cybersecurity",
    },
    "DJ_DP_.pdf": {
        "program_code": "6B03201",
        "program_slug": "dj",
        "program_name": "Digital Journalism",
        "school": "School of Creative Industries",
    },
    "DPA_DP_.pdf": {
        "program_code": "6B04104",
        "program_slug": "dpa",
        "program_name": "Digital Public Administration",
        "school": "School of Digital Public Administration",
    },
    "DTNPE_DP_.pdf": {
        "program_code": "6B07103",
        "program_slug": "dtnpe",
        "program_name": "Digital Technologies in Nuclear Power Engineering",
        "school": "School of Intelligent Systems",
    },
    "EE_DP_.pdf": {
        "program_code": "6B07102",
        "program_slug": "ee",
        "program_name": "Electronic Engineering",
        "school": "School of Intelligent Systems",
    },
    "IIOT_DP_.pdf": {
        "program_code": "6B07101",
        "program_slug": "iiot",
        "program_name": "Industrial Internet of Things",
        "school": "School of Intelligent Systems",
    },
    "ITE_DP_.pdf": {
        "program_code": "6B04102",
        "program_slug": "ite",
        "program_name": "IT Entrepreneurship",
        "school": "School of Creative Industries",
    },
    "ITM_DP_.pdf": {
        "program_code": "6B04101",
        "program_slug": "itm",
        "program_name": "IT Management",
        "school": "School of Creative Industries",
    },
    "IT_DP_.pdf": {
        "program_code": "6B06101",
        "program_slug": "it",
        "program_name": "Computer Science",
        "school": "School of Artificial Intelligence and Data Science",
    },
    "MCS_DP.pdf": {
        "program_code": "6B06107",
        "program_slug": "mcs",
        "program_name": "Mathematical and Computational Science",
        "school": "School of Artificial Intelligence and Data Science",
    },
    "MT_DP_.pdf": {
        "program_code": "6B06105",
        "program_slug": "mt",
        "program_name": "Media Technologies",
        "school": "School of Creative Industries",
    },
    "SE_DP_.pdf": {
        "program_code": "6B06102",
        "program_slug": "se",
        "program_name": "Software Engineering",
        "school": "School of Software Engineering",
    },
    "ST_DP_.pdf": {
        "program_code": "6B06202",
        "program_slug": "st",
        "program_name": "Smart Technologies",
        "school": "School of Intelligent Systems",
    },
}

PROGRAM_MARKDOWN_REGISTRY: dict[str, dict] = {
    "ai_business.md": {
        "program_code": "6B04103",
        "program_slug": "aib",
        "program_name": "AI Business",
        "school": "School of Creative Industries",
    },
    "big_data_analysis.md": {
        "program_code": "6B06103",
        "program_slug": "bda",
        "program_name": "Big Data Analysis",
        "school": "School of Artificial Intelligence and Data Science",
    },
    "it.md": {
        "program_code": "6B06101",
        "program_slug": "it",
        "program_name": "Computer Science",
        "school": "School of Artificial Intelligence and Data Science",
    },
    "cs.md": {
        "program_code": "6B06301",
        "program_slug": "cs",
        "program_name": "Cybersecurity",
        "school": "School of Cybersecurity",
    },
    "digital_journalism.md": {
        "program_code": "6B03201",
        "program_slug": "dj",
        "program_name": "Digital Journalism",
        "school": "School of Creative Industries",
    },
    "digital_public_administration.md": {
        "program_code": "6B04104",
        "program_slug": "dpa",
        "program_name": "Digital Public Administration",
        "school": "School of Digital Public Administration",
    },
    "digital_technologies_nuclear_power.md": {
        "program_code": "6B07103",
        "program_slug": "dtnpe",
        "program_name": "Digital Technologies in Nuclear Power Engineering",
        "school": "School of Intelligent Systems",
    },
    "electronic_engineering.md": {
        "program_code": "6B07102",
        "program_slug": "ee",
        "program_name": "Electronic Engineering",
        "school": "School of Intelligent Systems",
    },
    "industrial_internet_of_things.md": {
        "program_code": "6B07101",
        "program_slug": "iiot",
        "program_name": "Industrial Internet of Things",
        "school": "School of Intelligent Systems",
    },
    "it_entrepreneurship.md": {
        "program_code": "6B04102",
        "program_slug": "ite",
        "program_name": "IT Entrepreneurship",
        "school": "School of Creative Industries",
    },
    "it_management.md": {
        "program_code": "6B04101",
        "program_slug": "itm",
        "program_name": "IT Management",
        "school": "School of Creative Industries",
    },
    "mathematical_computational_science.md": {
        "program_code": "6B06107",
        "program_slug": "mcs",
        "program_name": "Mathematical and Computational Science",
        "school": "School of Artificial Intelligence and Data Science",
    },
    "media_technologies.md": {
        "program_code": "6B06105",
        "program_slug": "mt",
        "program_name": "Media Technologies",
        "school": "School of Creative Industries",
    },
    "smart_security_technologies.md": {
        "program_code": "6B06302",
        "program_slug": "sst",
        "program_name": "Smart Security Technologies",
        "school": "School of Cybersecurity",
    },
    "smart_technologies.md": {
        "program_code": "6B06202",
        "program_slug": "st",
        "program_name": "Smart Technologies",
        "school": "School of Intelligent Systems",
    },
    "software_engineering.md": {
        "program_code": "6B06102",
        "program_slug": "se",
        "program_name": "Software Engineering",
        "school": "School of Software Engineering",
    },
}

GENERAL_DOC_TYPES: dict[str, str] = {
    "academic_rules.md": "policy",
    "academic_calendar.md": "policy",
    "academic_mobility.md": "policy",
    "admissions.md": "handbook",
    "career_guide.md": "handbook",
    "contacts.md": "handbook",
    "digital_university.md": "handbook",
    "first_year_guide.md": "handbook",
    "general_faq.md": "faq",
    "library.md": "handbook",
    "military_department.md": "policy",
    "student_clubs.md": "handbook",
    "student_grants.md": "policy",
    "student_housing.md": "handbook",
    "student_life.md": "handbook",
    "student_support.md": "handbook",
    "tuition_and_discounts.md": "policy",
    "university_profile.md": "handbook",
}

GENERAL_SOURCE_TITLES: dict[str, str] = {
    "academic_rules.md": "Академические правила AITU",
    "academic_mobility.md": "Международная и академическая мобильность AITU",
    "admissions.md": "Поступление в AITU",
    "tuition_and_discounts.md": "Стоимость обучения и скидки AITU",
}

PROGRAM_ALIASES: dict[str, str] = {
    "aib": "6B04103",
    "ai business": "6B04103",
    "bda": "6B06103",
    "big data": "6B06103",
    "big data analysis": "6B06103",
    "бда": "6B06103",
    "биг дата": "6B06103",
    "se": "6B06102",
    "software engineering": "6B06102",
    "софтвер": "6B06102",
    "it": "6B06101",
    "computer science": "6B06101",
    "компьютерные науки": "6B06101",
    "cs": "6B06301",
    "cybersecurity": "6B06301",
    "cyber security": "6B06301",
    "кибербезопасность": "6B06301",
    "sst": "6B06302",
    "smart security technologies": "6B06302",
    "mcs": "6B06107",
    "mathematical and computational science": "6B06107",
    "mt": "6B06105",
    "media technologies": "6B06105",
    "dj": "6B03201",
    "digital journalism": "6B03201",
    "dpa": "6B04104",
    "digital public administration": "6B04104",
    "itm": "6B04101",
    "it management": "6B04101",
    "ite": "6B04102",
    "it entrepreneurship": "6B04102",
    "iiot": "6B07101",
    "industrial internet of things": "6B07101",
    "ee": "6B07102",
    "electronic engineering": "6B07102",
    "st": "6B06202",
    "smart technologies": "6B06202",
    "dtnpe": "6B07103",
    "digital technologies in nuclear power engineering": "6B07103",
}


def get_program_metadata(source: str | Path) -> dict:
    """Return stable programme metadata for a source path or filename."""
    name = Path(source).name
    if name in PROGRAM_MARKDOWN_REGISTRY:
        return PROGRAM_MARKDOWN_REGISTRY[name].copy()
    metadata = PROGRAM_REGISTRY.get(name, {}).copy()
    if metadata:
        return metadata

    stem = Path(source).stem.lower().replace("_dp", "").replace("_", "-")
    return {
        "program_code": None,
        "program_slug": stem or None,
        "program_name": Path(source).stem,
        "school": None,
    }


def get_markdown_metadata(source: str | Path) -> dict:
    """Infer stable metadata for curated Markdown documents."""
    path = Path(source)
    name = path.name
    normalized = path.as_posix().lower()

    metadata = {
        "source_format": "md",
        "source_group": "curated",
        "language": "ru",
        "priority": 90,
        "source_title": path.stem.replace("_", " ").title(),
        "program_code": None,
        "program_slug": None,
        "program_name": None,
        "school": None,
    }

    if "/programs/" in normalized or normalized.endswith("programs/" + name):
        if name == "index.md":
            metadata.update({
                "doc_type": "catalog",
                "source_title": "Educational Programs Catalog",
                "priority": 85,
            })
            return metadata
        program_metadata = PROGRAM_MARKDOWN_REGISTRY.get(name) or get_program_metadata(name)
        metadata.update(program_metadata)
        metadata.update({
            "doc_type": "program_profile",
            "source_title": program_metadata.get("program_name") or metadata["source_title"],
            "priority": 95,
            "updated_at": PROGRAM_SITE_REVIEWED_AT,
            "content_status": "current",
        })
        return metadata

    doc_type = GENERAL_DOC_TYPES.get(name, "handbook")
    metadata.update({
        "doc_type": doc_type,
        "priority": 80 if doc_type in {"policy", "faq"} else 75,
        "source_title": GENERAL_SOURCE_TITLES.get(name, metadata["source_title"]),
    })
    return metadata


def resolve_program_filter(value: str | None) -> dict:
    """Map legacy source filters and stable identifiers to programme metadata."""
    if not value:
        return {}
    normalized = value.strip()
    for source, metadata in PROGRAM_REGISTRY.items():
        candidates = {
            source,
            metadata.get("program_code"),
            metadata.get("program_slug"),
            metadata.get("program_name"),
        }
        if normalized in {str(candidate) for candidate in candidates if candidate}:
            resolved = metadata.copy()
            resolved["source"] = source
            return resolved
    for source, metadata in PROGRAM_MARKDOWN_REGISTRY.items():
        candidates = {
            source,
            metadata.get("program_code"),
            metadata.get("program_slug"),
            metadata.get("program_name"),
        }
        if normalized in {str(candidate) for candidate in candidates if candidate}:
            resolved = metadata.copy()
            resolved["source"] = source
            return resolved
    return {"source": normalized}


def resolve_program_mention(text: str | None) -> dict:
    """Apply an implicit filter only when exactly one programme is mentioned."""
    mentions = resolve_program_mentions(text)
    return mentions[0] if len(mentions) == 1 else {}


def resolve_program_mentions(text: str | None) -> list[dict]:
    """Recognize punctuation/code boundaries and avoid aliases inside longer names."""
    if not text:
        return []
    normalized = text.lower()
    aliases = dict(PROGRAM_ALIASES)
    for metadata in PROGRAM_MARKDOWN_REGISTRY.values():
        aliases[metadata["program_code"].lower()] = metadata["program_code"]
    matches = []
    codes = []
    for alias, code in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized):
            if any(start <= match.start() and match.end() <= end for start, end in matches):
                continue
            matches.append(match.span())
            if code not in codes:
                codes.append(code)
    return [resolve_program_filter(code) for code in codes]
