"""Contract types, program catalog adapter, and strict mock/real RAG wiring.

Per Point 1 of review:
- Only RAG_MODE=mock|real is accepted (default: 'mock').
- Unknown RAG_MODE values raise a configuration RuntimeError.
- When RAG_MODE='real', strictly requires backend.app.rag.pipeline.answer and raises
  RuntimeError if unavailable (never silently falls back to mock).
"""
from __future__ import annotations

import asyncio
import importlib
import os
import re
from collections.abc import AsyncIterator, Callable
from typing import Literal, TypedDict

from backend.app.api.pii import contains_pii
from backend.app.rag.programs import (
    PROGRAM_MARKDOWN_REGISTRY,
    resolve_program_filter,
    resolve_program_mentions,
)


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class RagStatusEvent(TypedDict):
    type: Literal["status"]
    stage: Literal["planning", "retrieving", "generating"]


class RagTokenEvent(TypedDict):
    type: Literal["token"]
    text: str


class RagDoneEvent(TypedDict):
    type: Literal["done"]
    finish_reason: Literal["stop", "guardrail", "no_data"]
    diagnostics: dict


class RagErrorEvent(TypedDict):
    type: Literal["error"]
    code: str
    message: str
    retryable: bool


RagEvent = RagStatusEvent | RagTokenEvent | RagDoneEvent | RagErrorEvent
AnswerFn = Callable[[str, list[HistoryMessage], str | None], AsyncIterator[RagEvent]]


_GUARANTEE_RE = re.compile(
    r"\b(?:гарантир\w*|100\s*%\s*(?:поступ\w*|грант\w*)|точно\s+поступ\w*|обеща\w+\s+грант)\b",
    re.I,
)
_THIRD_PARTY_PII_RE = re.compile(
    r"\b(?:личный\s+(?:номер|телефон|контакт)|телефон\s+(?:ректора|декана|преподавателя)|чуж\w+\s+(?:иин|данн\w+)|домашний\s+адрес)\b",
    re.I,
)


def list_programs() -> list[dict[str, str]]:
    """Return a stable, sorted list of educational programs for GET /api/programs."""
    programs_mod = importlib.import_module("backend.app.rag.programs")
    custom_fn = getattr(programs_mod, "list_programs", None)
    if callable(custom_fn):
        return list(custom_fn())

    items: dict[str, dict[str, str]] = {}
    for meta in PROGRAM_MARKDOWN_REGISTRY.values():
        code = meta.get("program_code")
        if not code or code in items:
            continue
        items[code] = {
            "program_code": str(code),
            "program_slug": str(meta.get("program_slug") or ""),
            "program_name": str(meta.get("program_name") or ""),
            "school": str(meta.get("school") or ""),
        }
    return sorted(items.values(), key=lambda item: (item["program_code"], item["program_name"]))


def is_valid_program_code(program_code: str | None) -> bool:
    if program_code is None:
        return True
    resolved = resolve_program_filter(program_code)
    return bool(resolved.get("program_code"))


def normalize_program_code(program_code: str | None) -> str | None:
    if not program_code:
        return None
    resolved = resolve_program_filter(program_code)
    return resolved.get("program_code")


async def mock_answer(
    question: str,
    history: list[HistoryMessage],
    program_code: str | None = None,
) -> AsyncIterator[RagEvent]:
    """Contract-compliant mock RAG generator for development and testing."""
    delay_ms = float(os.getenv("MOCK_RAG_STEP_MS", "15")) / 1000.0

    yield {"type": "status", "stage": "planning"}
    if delay_ms > 0:
        await asyncio.sleep(delay_ms)

    # Guardrails: admission guarantees, personal data, private contacts
    if contains_pii(question) or _GUARANTEE_RE.search(question) or _THIRD_PARTY_PII_RE.search(question):
        yield {"type": "status", "stage": "generating"}
        guardrail_text = (
            "Я не обрабатываю персональные данные, личные контакты сотрудников/студентов "
            "и не даю индивидуальных гарантий поступления или присуждения гранта. "
            "По официальным вопросам приёма обратитесь в приёмную комиссию Astana IT University: "
            "admission@astanait.edu.kz или +7 (7172) 645-710."
        )
        for chunk in _chunk_text(guardrail_text):
            if delay_ms > 0:
                await asyncio.sleep(delay_ms)
            yield {"type": "token", "text": chunk}
        yield {
            "type": "done",
            "finish_reason": "guardrail",
            "diagnostics": {
                "mode": "mock",
                "guardrail": True,
                "history_used": len(history),
            },
        }
        return

    yield {"type": "status", "stage": "retrieving"}
    if delay_ms > 0:
        await asyncio.sleep(delay_ms)

    canonical_code = normalize_program_code(program_code)
    mentioned = resolve_program_mentions(question)
    if not canonical_code and len(mentioned) == 1:
        canonical_code = mentioned[0].get("program_code")

    program_info = resolve_program_filter(canonical_code) if canonical_code else {}
    program_label = (
        f"**{program_info['program_name']} ({program_info['program_code']})**"
        if program_info.get("program_name")
        else "образовательным программам **Astana IT University**"
    )

    history_note = (
        f" *(с учётом контекста из {len(history)} пред. сообщ.)*"
        if history
        else ""
    )

    response_text = (
        f"Ответ по {program_label}{history_note}:\n\n"
        f"- **Ваш вопрос:** {question.strip()}\n"
        f"- **Срок обучения на бакалавриате:** 3 года (триместровая система).\n"
        f"- **Язык обучения:** английский.\n"
        f"- **Официальные контакты:** admission@astanait.edu.kz, +7 (7172) 645-710."
    )

    yield {"type": "status", "stage": "generating"}
    for chunk in _chunk_text(response_text):
        if delay_ms > 0:
            await asyncio.sleep(delay_ms)
        yield {"type": "token", "text": chunk}

    yield {
        "type": "done",
        "finish_reason": "stop",
        "diagnostics": {
            "mode": "mock",
            "program_code": canonical_code,
            "history_used": len(history),
            "sources": [program_info.get("source") or "SE_DP_.pdf"],
        },
    }


def _chunk_text(text: str, words_per_chunk: int = 4) -> list[str]:
    tokens = re.findall(r"\S+\s*", text)
    if not tokens:
        return [text]
    return [
        "".join(tokens[i : i + words_per_chunk])
        for i in range(0, len(tokens), words_per_chunk)
    ]


def get_answer_fn() -> AnswerFn:
    """Resolve the RAG answer generator strictly according to RAG_MODE ('mock' or 'real')."""
    mode = os.getenv("RAG_MODE", "mock").strip().lower()
    if mode == "mock":
        return mock_answer

    if mode != "real":
        raise RuntimeError(f"Неизвестное значение RAG_MODE='{mode}'. Ожидается 'mock' или 'real'.")

    last_error: Exception | None = None
    for module_name in ("backend.app.rag.pipeline", "backend.app.rag"):
        try:
            mod = importlib.import_module(module_name)
            real_fn = getattr(mod, "answer", None)
            if callable(real_fn):
                return real_fn
        except Exception as exc:
            last_error = exc

    detail = f" ({last_error})" if last_error else ""
    raise RuntimeError(
        "RAG_MODE='real' установлен, но функция 'async def answer(question, history, program_code=None)' "
        f"не найдена в backend.app.rag.pipeline{detail}"
    )
