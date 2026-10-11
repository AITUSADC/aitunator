"""Shared query processing for live retrieval and offline measurements."""
import json
import logging
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from backend.app.rag.programs import resolve_program_filter, resolve_program_mention
logger = logging.getLogger(__name__)

_ru_stemmer = SnowballStemmer("russian")
_en_stemmer = SnowballStemmer("english")
try:
    _ru_stops = set(stopwords.words("russian"))
    _en_stops = set(stopwords.words("english"))
except LookupError:
    logger.warning("NLTK stopwords corpus is unavailable; BM25 tokenization will keep stopwords")
    _ru_stops = set()
    _en_stops = set()



_ELECTIVE_QUERY_RE = re.compile(r"\b(электив\w*|выборн\w* дисциплин|компонент\w* по выбору|elective\w*)\b", re.IGNORECASE)
_GENERAL_MOBILITY_RE = re.compile(
    r"\b(double\s*degree|двойн\w* диплом\w*|академическ\w* мобильност\w*|academic mobility)\b",
    re.IGNORECASE,
)

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text.lower())
    result = []
    for token in tokens:
        if token in _ru_stops or token in _en_stops:
            continue
        if re.search(r"[а-яё]", token):
            result.append(_ru_stemmer.stem(token))
        else:
            result.append(_en_stemmer.stem(token))
    return result


def _rrf(rankings: list[list], k: int = 60) -> list:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _normalize_scope(
    source_filter: str | None = None,
    program_code: str | None = None,
    doc_types: tuple[str, ...] | None = None,
    language: str | None = None,
) -> dict:
    resolved = resolve_program_filter(source_filter)
    return {
        "source_filter": source_filter,
        "source": resolved.get("source"),
        "program_code": program_code or resolved.get("program_code"),
        "program_slug": resolved.get("program_slug"),
        "doc_types": tuple(doc_types or ()),
        "language": language,
    }


def _scope_cache_key(scope: dict) -> str:
    return json.dumps(scope, ensure_ascii=False, sort_keys=True)


def _apply_query_program_scope(scope: dict, query: str) -> dict:
    if scope.get("program_code") or scope.get("program_slug") or scope.get("source"):
        return scope
    resolved = resolve_program_mention(query)
    if not resolved:
        return scope
    scoped = dict(scope)
    scoped["source"] = resolved.get("source")
    scoped["program_code"] = resolved.get("program_code")
    scoped["program_slug"] = resolved.get("program_slug")
    return scoped


def _expand_query_aliases(query: str, scope: dict) -> str:
    additions: list[str] = []
    if scope.get("program_code") == "6B06103" and "big data analysis" not in query.lower():
        additions.append("Big Data Analysis BDA 6B06103")
    if re.search(r"\b(предмет|предметы|изучают|учат|обучаются)\b", query, re.IGNORECASE):
        additions.append("дисциплины учебный план основные дисциплины curriculum courses")
    if _ELECTIVE_QUERY_RE.search(query):
        additions.append("элективные дисциплины компонент по выбору elective courses")
    if additions:
        return f"{query}\n{' '.join(additions)}"
    return query


def _payload_matches_scope(payload: dict, scope: dict) -> bool:
    if scope.get("language") and payload.get("language") != scope["language"]:
        return False
    if scope.get("doc_types") and payload.get("doc_type") not in scope["doc_types"]:
        return False

    selectors = [scope.get("source"), scope.get("program_code"), scope.get("program_slug")]
    selectors = [selector for selector in selectors if selector]
    if not selectors:
        return True
    return (
        payload.get("source") in selectors
        or payload.get("program_code") in selectors
        or payload.get("program_slug") in selectors
    )


def _detect_query_section_type(query: str) -> str | None:
    q = query.lower()
    if re.search(r"карьер|работ|трудоустрой|кем|професс", q):
        return "career"
    if _ELECTIVE_QUERY_RE.search(q) or re.search(r"дисциплин|курс|предмет|модул|curriculum", q):
        return "curriculum"
    if re.search(r"навык|умени|компетенц|learn|skill", q):
        return "skills"
    if re.search(r"кому|подходит|для кого|fit", q):
        return "fit"
    if re.search(r"поступ|admission|документ", q):
        return "admission"
    return None


def _metadata_boost(query: str, result: dict, scope: dict) -> float:
    boost = 0.0
    if result.get("source_group") == "curated":
        boost += 0.12
    if result.get("doc_type") == "program_profile":
        boost += 0.08
    if scope.get("program_code") and result.get("program_code") == scope["program_code"]:
        boost += 0.10
    if scope.get("program_slug") and result.get("program_slug") == scope["program_slug"]:
        boost += 0.08

    section_type = _detect_query_section_type(query)
    if section_type and result.get("section_type") == section_type:
        boost += 0.10
    if _ELECTIVE_QUERY_RE.search(query):
        section = str(result.get("section") or "").lower()
        if section.startswith("электив"):
            boost += 0.30
    if result.get("source_group") == "raw" and result.get("type") == "table":
        boost -= 0.06
    boost += min(float(result.get("priority") or 0), 100.0) / 1000.0
    return boost

