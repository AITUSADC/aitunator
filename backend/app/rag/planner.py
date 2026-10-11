"""Deterministic multi-program query planning; LLM/history planning is not implemented."""
import re
from backend.app.rag.programs import PROGRAM_ALIASES, resolve_program_mentions


def program_queries(query: str) -> list[tuple[str, str]]:
    """Fan out explicit multi-program questions without reading evaluation labels."""
    programs = resolve_program_mentions(query)
    if len(programs) < 2:
        return []
    aliases = set(PROGRAM_ALIASES)
    aliases.update(p["program_code"].lower() for p in programs)
    # One alternation prevents short aliases inside already matched long names.
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?!\w)"
    topic = re.sub(pattern, " ", query, flags=re.I)
    topic = re.sub(r"\b(?:сравни\w*|compare|versus|vs|каждой|обеих)\b", " ", topic, flags=re.I)
    topic = re.sub(r"\s+", " ", topic).strip(" ,:-")
    # Canonical code order makes selection independent of alias string length.
    return [(p["program_code"], f"{p['program_name']} {p['program_code']}. {topic}")
            for p in sorted(programs, key=lambda p: p["program_code"])]


