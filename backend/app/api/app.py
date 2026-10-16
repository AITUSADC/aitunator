"""FastAPI application implementing the AITUNATOR v1 API and SSE contract."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.api.db import Database, HISTORY_WINDOW
from backend.app.api.pii import contains_pii
from backend.app.api.rag_bridge import (
    AnswerFn,
    get_answer_fn,
    is_valid_program_code,
    list_programs,
    normalize_program_code,
)
from backend.app.api.security import (
    SessionLockRegistry,
    SlidingWindowRateLimiter,
    get_client_ip,
)
from backend.app.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PARSER_VERSION,
    QDRANT_URL,
)
from backend.app.rag.programs import PROGRAM_SITE_REVIEWED_AT

logger = logging.getLogger("aitunator.api")

MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
MAX_COMMENT_LENGTH = int(os.getenv("MAX_COMMENT_LENGTH", "1000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
TTFT_TIMEOUT_SECONDS = float(os.getenv("TTFT_TIMEOUT_SECONDS", "15.0"))
TOTAL_TIMEOUT_SECONDS = float(os.getenv("TOTAL_TIMEOUT_SECONDS", "60.0"))
DISCONNECT_POLL_SECONDS = float(os.getenv("DISCONNECT_POLL_SECONDS", "0.2"))


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    session_id: Union[str, None] = Field(default=None, max_length=128)
    program_code: Union[str, None] = Field(default=None, max_length=64)


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1, max_length=128)
    rating: Union[int, str]
    comment: Union[str, None] = Field(default=None, max_length=MAX_COMMENT_LENGTH)


def _normalize_chat_request(payload: ChatRequest) -> tuple[str, str | None, str | None]:
    question = (payload.question or "").strip()
    if not question:
        raise ValueError("Вопрос не может быть пустым.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Вопрос не должен превышать {MAX_QUESTION_LENGTH} символов.")

    session_id = payload.session_id.strip() if payload.session_id else None
    session_id = session_id or None

    raw_prog = payload.program_code.strip() if payload.program_code else None
    if raw_prog:
        if not is_valid_program_code(raw_prog):
            raise ValueError(f"Неизвестный код программы: {raw_prog}")
        program_code = normalize_program_code(raw_prog)
    else:
        program_code = None

    return question, session_id, program_code


def _normalize_feedback_request(payload: FeedbackRequest) -> tuple[str, int, str | None]:
    message_id = (payload.message_id or "").strip()
    if not message_id:
        raise ValueError("Не указан message_id.")

    val = payload.rating
    if isinstance(val, int):
        if val not in (-1, 1):
            raise ValueError("Оценка должна быть 1 или -1.")
        rating = val
    else:
        norm = str(val).strip().lower()
        if norm in ("up", "like", "1", "+1"):
            rating = 1
        elif norm in ("down", "dislike", "-1"):
            rating = -1
        else:
            raise ValueError("Оценка должна быть 1 (up) или -1 (down).")

    comment = payload.comment.strip() if payload.comment else None
    comment = comment or None
    if comment and len(comment) > MAX_COMMENT_LENGTH:
        raise ValueError(f"Комментарий не должен превышать {MAX_COMMENT_LENGTH} символов.")

    return message_id, rating, comment


def _format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _probe_qdrant_sync(qdrant_url: str, collection_name: str, timeout: float = 1.5) -> dict[str, Any]:
    base = qdrant_url.rstrip("/")
    info: dict[str, Any] = {
        "available": False,
        "url": base,
        "collection": collection_name,
        "collection_exists": False,
        "points_count": None,
    }
    try:
        req = urllib.request.Request(f"{base}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                info["available"] = True
        col_req = urllib.request.Request(f"{base}/collections/{collection_name}", method="GET")
        with urllib.request.urlopen(col_req, timeout=timeout) as col_resp:
            if col_resp.status == 200:
                body = json.loads(col_resp.read().decode("utf-8"))
                result = body.get("result") or {}
                info["collection_exists"] = True
                info["points_count"] = result.get("points_count")
    except Exception:
        pass
    return info


def create_app(
    *,
    db_path: str | Path | None = None,
    answer_fn: AnswerFn | None = None,
    rate_limit_per_minute: int | None = None,
    ttft_timeout: float | None = None,
    total_timeout: float | None = None,
    frontend_dist: str | Path | None = None,
) -> FastAPI:
    resolved_db = Database(db_path=db_path)
    resolved_limiter = SlidingWindowRateLimiter(
        max_requests=(
            rate_limit_per_minute
            if rate_limit_per_minute is not None
            else int(os.getenv("RATE_LIMIT_PER_MINUTE", str(RATE_LIMIT_PER_MINUTE)))
        ),
        window_seconds=60.0,
    )
    resolved_locks = SessionLockRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleaned = await app.state.db.init()
        logger.info("SQLite initialized (expired sessions cleaned: %d)", cleaned)

        # Track RAG readiness state per Point 2 of review
        mode = os.getenv("RAG_MODE", "mock").strip().lower()
        if mode not in ("mock", "real"):
            app.state.rag_ready = False
            app.state.rag_error = f"Неизвестное значение RAG_MODE='{mode}'. Ожидается 'mock' или 'real'."
        elif mode == "mock":
            app.state.rag_ready = True
            app.state.rag_error = None
        else:
            # Mode is 'real': execute optional startup hook and verify answer() is present
            try:
                import importlib
                pipe_mod = importlib.import_module("backend.app.rag.pipeline")
                startup_fn = getattr(pipe_mod, "startup", None) or getattr(pipe_mod, "init_rag", None)
                if callable(startup_fn):
                    res = startup_fn()
                    if asyncio.iscoroutine(res):
                        await res
                real_answer = getattr(pipe_mod, "answer", None)
                if not callable(real_answer):
                    raise RuntimeError("Функция 'answer' не найдена в backend.app.rag.pipeline")
                app.state.rag_ready = True
                app.state.rag_error = None
            except Exception as exc:
                logger.exception("Failed to initialize real RAG in lifespan")
                app.state.rag_ready = False
                app.state.rag_error = f"Ошибка инициализации RAG-пайплайна: {exc}"

        yield

        # Lifespan shutdown
        if os.getenv("RAG_MODE", "mock").strip().lower() == "real":
            try:
                import importlib
                pipe_mod = importlib.import_module("backend.app.rag.pipeline")
                shutdown_fn = getattr(pipe_mod, "shutdown", None)
                if callable(shutdown_fn):
                    res = shutdown_fn()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass

    app = FastAPI(
        title="AITUNATOR API",
        version="0.1.0",
        description="Astana IT University RAG Assistant API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.db = resolved_db
    app.state.rate_limiter = resolved_limiter
    app.state.session_locks = resolved_locks
    app.state.answer_fn = answer_fn
    app.state.rag_ready = True
    app.state.rag_error = None
    app.state.ttft_timeout = (
        ttft_timeout
        if ttft_timeout is not None
        else float(os.getenv("TTFT_TIMEOUT_SECONDS", str(TTFT_TIMEOUT_SECONDS)))
    )
    app.state.total_timeout = (
        total_timeout
        if total_timeout is not None
        else float(os.getenv("TOTAL_TIMEOUT_SECONDS", str(TOTAL_TIMEOUT_SECONDS)))
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first_msg = errors[0].get("msg", "Некорректные параметры запроса.") if errors else "Некорректные параметры запроса."
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": first_msg,
            },
        )

    @app.post("/api/chat")
    async def chat_endpoint(payload: ChatRequest, request: Request):
        mode = os.getenv("RAG_MODE", "mock").strip().lower()

        # Reject unknown RAG_MODE explicitly per Point 1
        if mode not in ("mock", "real"):
            return JSONResponse(
                status_code=500,
                content={
                    "code": "config_error",
                    "message": f"Неизвестное значение RAG_MODE='{mode}'. Допустимы только 'mock' или 'real'.",
                    "retryable": False,
                },
            )

        # Check RAG readiness state per Point 2
        if not getattr(app.state, "rag_ready", True) and mode == "real":
            return JSONResponse(
                status_code=503,
                content={
                    "code": "rag_not_ready",
                    "message": app.state.rag_error or "Реальный RAG-пайплайн не готов к обработке запросов.",
                    "retryable": False,
                },
            )

        try:
            question, req_session_id, program_code = _normalize_chat_request(payload)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"code": "validation_error", "message": str(exc)},
            )

        client_ip = get_client_ip(request)
        allowed, retry_after = app.state.rate_limiter.allow(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "code": "rate_limit",
                    "message": "Превышен лимит запросов (не более 10 запросов в минуту). Попробуйте чуть позже.",
                    "retryable": True,
                },
            )

        session_id = req_session_id or str(uuid.uuid4())
        if not app.state.session_locks.try_acquire(session_id):
            return JSONResponse(
                status_code=409,
                content={
                    "code": "session_busy",
                    "message": "В этой сессии уже выполняется запрос. Дождитесь ответа или нажмите «Стоп».",
                    "retryable": True,
                },
            )

        try:
            history = await app.state.db.get_recent_history(session_id, limit=HISTORY_WINDOW)
            rag_runner: AnswerFn = app.state.answer_fn or get_answer_fn()
        except RuntimeError as exc:
            app.state.session_locks.release(session_id)
            return JSONResponse(
                status_code=503,
                content={
                    "code": "rag_not_ready",
                    "message": str(exc),
                    "retryable": False,
                },
            )
        except Exception:
            app.state.session_locks.release(session_id)
            raise

        async def event_stream():
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def _produce() -> None:
                try:
                    async for ev in rag_runner(question, history, program_code):
                        await queue.put(dict(ev))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Unhandled error inside RAG pipeline (session_id=%s)", session_id)
                    await queue.put(
                        {
                            "type": "error",
                            "code": "rag_internal_error",
                            "message": "Произошла внутренняя ошибка при генерации ответа.",
                            "retryable": True,
                        }
                    )
                finally:
                    await queue.put(None)

            producer_task = asyncio.create_task(_produce())
            started_at = time.monotonic()
            first_token_seen = False
            terminal_event_seen = False
            answer_chunks: list[str] = []
            ttft_limit: float = app.state.ttft_timeout
            total_limit: float = app.state.total_timeout

            try:
                while True:
                    if await request.is_disconnected():
                        producer_task.cancel()
                        try:
                            await producer_task
                        except asyncio.CancelledError:
                            pass
                        break

                    now = time.monotonic()
                    elapsed = now - started_at
                    remaining_total = total_limit - elapsed
                    if remaining_total <= 0:
                        producer_task.cancel()
                        try:
                            await producer_task
                        except asyncio.CancelledError:
                            pass
                        terminal_event_seen = True
                        yield _format_sse(
                            "error",
                            {
                                "code": "timeout",
                                "message": "Превышено максимальное время генерации ответа (60 секунд).",
                                "retryable": True,
                            },
                        )
                        break

                    if not first_token_seen:
                        remaining_ttft = ttft_limit - elapsed
                        if remaining_ttft <= 0:
                            producer_task.cancel()
                            try:
                                await producer_task
                            except asyncio.CancelledError:
                                pass
                            terminal_event_seen = True
                            yield _format_sse(
                                "error",
                                {
                                    "code": "timeout",
                                    "message": "Превышено время ожидания первого токена ответа (15 секунд).",
                                    "retryable": True,
                                },
                            )
                            break
                        step_timeout = min(DISCONNECT_POLL_SECONDS, remaining_ttft, remaining_total)
                    else:
                        step_timeout = min(DISCONNECT_POLL_SECONDS, remaining_total)

                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=max(0.005, step_timeout))
                    except asyncio.TimeoutError:
                        continue

                    if item is None:
                        # Producer finished. Check if a terminal event (done or error) was sent per Point 4!
                        if not terminal_event_seen:
                            yield _format_sse(
                                "error",
                                {
                                    "code": "incomplete_stream",
                                    "message": "Поток генерации неожиданно завершился без подтверждения завершения.",
                                    "retryable": True,
                                },
                            )
                        break

                    ev_type = item.get("type")
                    if ev_type == "status":
                        stage = str(item.get("stage") or "planning")
                        yield _format_sse("status", {"stage": stage})

                    elif ev_type == "token":
                        text = str(item.get("text") or "")
                        if text:
                            first_token_seen = True
                            answer_chunks.append(text)
                            yield _format_sse("token", {"text": text})

                    elif ev_type == "error":
                        terminal_event_seen = True
                        yield _format_sse(
                            "error",
                            {
                                "code": str(item.get("code") or "rag_error"),
                                "message": str(
                                    item.get("message") or "Не удалось сформировать ответ."
                                ),
                                "retryable": bool(item.get("retryable", True)),
                            },
                        )
                        break

                    elif ev_type == "done":
                        terminal_event_seen = True
                        finish_reason = str(item.get("finish_reason") or "stop")
                        diagnostics = item.get("diagnostics") or {}
                        full_answer = "".join(answer_chunks)

                        # Check PII with strict official contacts rule (Point 3)
                        skip_save = (
                            finish_reason == "guardrail"
                            or contains_pii(question)
                            or contains_pii(full_answer, allow_official_contacts=True)
                        )

                        message_id: str | None = None
                        if not skip_save:
                            try:
                                message_id = await app.state.db.save_completed_turn(
                                    session_id=session_id,
                                    question=question,
                                    answer_text=full_answer,
                                    program_code=program_code,
                                    finish_reason=finish_reason,
                                    diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to commit chat turn to SQLite (session_id=%s)",
                                    session_id,
                                )
                                yield _format_sse(
                                    "error",
                                    {
                                        "code": "storage_error",
                                        "message": "Не удалось сохранить сообщение в базу данных.",
                                        "retryable": True,
                                    },
                                )
                                break

                        yield _format_sse(
                            "done",
                            {
                                "session_id": session_id,
                                "message_id": message_id,
                                "finish_reason": finish_reason,
                            },
                        )
                        break

            except asyncio.CancelledError:
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
                raise
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                    try:
                        await producer_task
                    except asyncio.CancelledError:
                        pass
                app.state.session_locks.release(session_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-Id": session_id,
            },
        )

    @app.get("/api/sessions/{session_id}")
    async def get_session_endpoint(session_id: str):
        cleaned = session_id.strip()
        if not cleaned or len(cleaned) > 128:
            raise HTTPException(status_code=400, detail="Некорректный идентификатор сессии.")
        session_data = await app.state.db.get_session_history(cleaned)
        if session_data is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена или истёк срок хранения.")
        return session_data

    @app.post("/api/feedback")
    async def feedback_endpoint(payload: FeedbackRequest):
        try:
            message_id, rating, comment = _normalize_feedback_request(payload)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"code": "validation_error", "message": str(exc)},
            )
        if comment and contains_pii(comment):
            return JSONResponse(
                status_code=422,
                content={
                    "code": "pii_detected",
                    "message": "Комментарий содержит персональные данные и не может быть сохранён.",
                },
            )
        try:
            saved = await app.state.db.save_feedback(
                message_id=message_id,
                rating=rating,
                comment=comment,
            )
            return saved
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="Сообщение не найдено (возможно, оно не было сохранено в истории).",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/programs")
    async def programs_endpoint():
        programs = list_programs()
        return {
            "programs": programs,
            "count": len(programs),
            "reviewed_at": PROGRAM_SITE_REVIEWED_AT,
        }

    @app.get("/api/health")
    async def health_endpoint():
        db_ok, journal_mode = await app.state.db.check_health()
        qdrant_info = await asyncio.to_thread(
            _probe_qdrant_sync, QDRANT_URL, COLLECTION_NAME, 1.5
        )
        rag_mode = os.getenv("RAG_MODE", "mock").strip().lower()
        rag_ready = bool(getattr(app.state, "rag_ready", True))
        rag_error = getattr(app.state, "rag_error", None)

        status = "ok" if (db_ok and (rag_mode != "real" or rag_ready)) else "degraded"

        embedder_name = os.getenv("EMBEDDER") or os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
        llm_provider = os.getenv("LLM_PROVIDER", "groq")
        llm_model = os.getenv("LLM_MODEL") or os.getenv("LLM_MAIN_MODEL", "llama-3.3-70b-versatile")
        reranker_name = os.getenv("RERANKER") or os.getenv("RERANK_PROVIDER", "noop")

        return {
            "status": status,
            "sqlite": {
                "available": db_ok,
                "journal_mode": journal_mode,
            },
            "rag": {
                "mode": rag_mode,
                "ready": rag_ready,
                "error": rag_error,
            },
            "qdrant": qdrant_info,
            "version": {
                "kb_version": os.getenv("KB_VERSION", PARSER_VERSION),
                "parser_version": PARSER_VERSION,
                "programs_reviewed_at": PROGRAM_SITE_REVIEWED_AT,
                "embedding_model": embedder_name,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "reranker": reranker_name,
            },
        }

    dist_dir = Path(
        frontend_dist
        or os.getenv(
            "FRONTEND_DIST_DIR",
            str(Path(__file__).resolve().parents[3] / "frontend" / "dist"),
        )
    )
    if dist_dir.is_dir() and (dist_dir / "index.html").is_file():
        assets_dir = dist_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def _serve_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            candidate = (dist_dir / full_path).resolve()
            if (
                full_path
                and candidate.is_relative_to(dist_dir.resolve())
                and candidate.is_file()
            ):
                return FileResponse(str(candidate))
            return FileResponse(str(dist_dir / "index.html"))

    return app


app = create_app()
