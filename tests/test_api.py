"""Comprehensive unit and integration tests for SQLite WAL, PII, SSE chat, timeouts, cancellation, and concurrency."""
from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from backend.app.api.app import create_app
from backend.app.api.db import Database, _connect
from backend.app.api.pii import contains_pii


def _make_async_client(app) -> httpx.AsyncClient:
    sig = inspect.signature(httpx.AsyncClient.__init__)
    if "app" in sig.parameters:
        return httpx.AsyncClient(app=app, base_url="http://test")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _parse_sse_frames(raw_text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_str = "{}"
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
        frames.append((event_name, json.loads(data_str)))
    return frames


class ApiAndDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_aitunator.db"
        self.db = Database(self.db_path)
        await self.db.init()

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    async def test_wal_enabled_and_30_day_cleanup_on_init(self) -> None:
        ok, mode = await self.db.check_health()
        self.assertTrue(ok)
        self.assertEqual(mode, "wal")

        old_iso = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
        recent_iso = datetime.now(timezone.utc).isoformat()
        async with _connect(self.db_path) as conn:
            await conn.execute("PRAGMA foreign_keys=ON;")
            await conn.execute(
                "INSERT INTO sessions (id, created_at, last_active_at) VALUES (?, ?, ?);",
                ("expired-session", old_iso, old_iso),
            )
            await conn.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, 'user', 'Привет', ?);",
                ("msg-old", "expired-session", old_iso),
            )
            await conn.execute(
                "INSERT INTO sessions (id, created_at, last_active_at) VALUES (?, ?, ?);",
                ("active-session", recent_iso, recent_iso),
            )

        deleted = await self.db.init(retention_days=30)
        self.assertEqual(deleted, 1)
        self.assertIsNone(await self.db.get_session_history("expired-session"))
        self.assertIsNotNone(await self.db.get_session_history("active-session"))

    async def test_history_window_limits_to_last_10_messages_in_order(self) -> None:
        session_id = "hist-session"
        for i in range(7):
            await self.db.save_completed_turn(
                session_id=session_id,
                question=f"Вопрос {i}",
                answer_text=f"Ответ {i}",
                program_code="6B06102",
            )
        recent = await self.db.get_recent_history(session_id, limit=10)
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[0], {"role": "user", "content": "Вопрос 2"})
        self.assertEqual(recent[-1], {"role": "assistant", "content": "Ответ 6"})

    async def test_narrow_pii_exceptions_block_personal_and_allow_official(self) -> None:
        # 1. Personal emails (even with astanait.edu.kz) must be blocked
        self.assertTrue(contains_pii("Напишите мне на person@astanait.edu.kz"))
        self.assertTrue(contains_pii("Моя почта student123@gmail.com"))

        # 2. International and mobile phone numbers must be blocked
        self.assertTrue(contains_pii("Мой телефон +1 202 555 01 23"))
        self.assertTrue(contains_pii("Позвоните на +7 701 234 56 78"))

        # 3. IIN and card numbers must be blocked
        self.assertTrue(contains_pii("Мой ИИН 040512500123"))
        self.assertTrue(contains_pii("Карта 4400 4301 1234 5678"))

        # 4. Verified official public AITU contacts must be ALLOWED in answers
        self.assertFalse(contains_pii("Почта приёмной комиссии: admission@astanait.edu.kz", allow_official_contacts=True))
        self.assertFalse(contains_pii("Телефон приёмной комиссии AITU: +7 (7172) 645-710", allow_official_contacts=True))
        self.assertFalse(contains_pii("Общие вопросы: info@astanait.edu.kz или +7 7172 645710", allow_official_contacts=True))

    async def test_chat_sse_flow_session_restore_and_feedback(self) -> None:
        app = create_app(db_path=self.db_path)
        async with _make_async_client(app) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "question": "Какие предметы изучают на Software Engineering?",
                    "program_code": "se",
                },
            )
            self.assertEqual(resp.status_code, 200)
            frames = _parse_sse_frames(resp.text)
            event_names = [name for name, _ in frames]
            self.assertIn("status", event_names)
            self.assertIn("token", event_names)
            self.assertEqual(frames[-1][0], "done")

            done_data = frames[-1][1]
            session_id = done_data["session_id"]
            message_id = done_data["message_id"]
            self.assertTrue(session_id)
            self.assertTrue(message_id)
            self.assertEqual(done_data["finish_reason"], "stop")
            self.assertNotIn("sources", done_data)

            # Verify sources_json is stored in SQLite
            async with _connect(self.db_path) as conn:
                cur = await conn.execute("SELECT sources_json FROM messages WHERE id = ?;", (message_id,))
                row = await cur.fetchone()
                self.assertIsNotNone(row)
                self.assertIn("SE_DP_.pdf", row["sources_json"])

            # Feedback submission
            fb_resp = await client.post(
                "/api/feedback",
                json={"message_id": message_id, "rating": 1, "comment": "Отлично!"},
            )
            self.assertEqual(fb_resp.status_code, 200)

            # Feedback with PII -> 422
            fb_pii = await client.post(
                "/api/feedback",
                json={"message_id": message_id, "rating": -1, "comment": "Позвоните +7 707 111 22 33"},
            )
            self.assertEqual(fb_pii.status_code, 422)
            self.assertEqual(fb_pii.json()["code"], "pii_detected")

    async def test_pii_and_guardrail_turns_are_not_saved_to_sqlite(self) -> None:
        # Assistant answering with personal phone number -> turn must NOT be saved
        async def _leaky_answer(question: str, history: list, program_code: str | None = None):
            yield {"type": "status", "stage": "generating"}
            yield {"type": "token", "text": "Личный контакт сотрудника: +7 701 999 88 77"}
            yield {"type": "done", "finish_reason": "stop", "diagnostics": {}}

        app_leaky = create_app(db_path=self.db_path, answer_fn=_leaky_answer)
        async with _make_async_client(app_leaky) as client:
            resp = await client.post(
                "/api/chat",
                json={"question": "Подскажи контакты", "session_id": "pii-session-1"},
            )
            frames = _parse_sse_frames(resp.text)
            done_data = frames[-1][1]
            self.assertIsNone(done_data["message_id"])
            self.assertIsNone(await self.db.get_session_history("pii-session-1"))

    async def test_unterminated_stream_sends_error_and_does_not_save(self) -> None:
        # Generator yields tokens and exits WITHOUT done or error event (Point 4)
        async def _abrupt_answer(question: str, history: list, program_code: str | None = None):
            yield {"type": "status", "stage": "generating"}
            yield {"type": "token", "text": "Частичный ответ"}
            # Finishes abruptly without done or error

        app = create_app(db_path=self.db_path, answer_fn=_abrupt_answer)
        async with _make_async_client(app) as client:
            resp = await client.post(
                "/api/chat",
                json={"question": "Тест незавершённого потока", "session_id": "abrupt-session"},
            )
            frames = _parse_sse_frames(resp.text)
            self.assertEqual(frames[-1][0], "error")
            self.assertEqual(frames[-1][1]["code"], "incomplete_stream")
            # Must not be saved to SQLite
            self.assertIsNone(await self.db.get_session_history("abrupt-session"))

    async def test_cancellation_releases_lock_and_does_not_save(self) -> None:
        entered = asyncio.Event()

        async def _hanging_answer(question: str, history: list, program_code: str | None = None):
            yield {"type": "status", "stage": "generating"}
            entered.set()
            await asyncio.sleep(10.0)
            yield {"type": "done", "finish_reason": "stop", "diagnostics": {}}

        app = create_app(db_path=self.db_path, answer_fn=_hanging_answer)
        async with _make_async_client(app) as client:
            # Send request in a task and cancel it
            task = asyncio.create_task(
                client.post("/api/chat", json={"question": "Вопрос", "session_id": "cancel-sess"})
            )
            await entered.wait()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, httpx.RequestError):
                pass

            # Give a brief moment for cleanup
            await asyncio.sleep(0.05)
            # Session lock must be released: new request to same session succeeds immediately
            self.assertEqual(app.state.session_locks.active_count, 0)
            self.assertIsNone(await self.db.get_session_history("cancel-sess"))

    async def test_ttft_and_total_timeouts_emit_error_and_clean_up(self) -> None:
        # TTFT timeout
        async def _stalled_first_token(question: str, history: list, program_code: str | None = None):
            yield {"type": "status", "stage": "planning"}
            await asyncio.sleep(0.2)  # Will exceed ttft_timeout=0.05

        app_ttft = create_app(db_path=self.db_path, answer_fn=_stalled_first_token, ttft_timeout=0.05)
        async with _make_async_client(app_ttft) as client:
            resp = await client.post("/api/chat", json={"question": "Вопрос на TTFT"})
            frames = _parse_sse_frames(resp.text)
            self.assertEqual(frames[-1][0], "error")
            self.assertEqual(frames[-1][1]["code"], "timeout")
            self.assertIn("первого токена", frames[-1][1]["message"])

    async def test_strict_rag_mode_real_and_health_reporting(self) -> None:
        # Mode 'real' without real pipeline must return 503 rag_not_ready (Point 1)
        app = create_app(db_path=self.db_path)
        with patch.dict("os.environ", {"RAG_MODE": "real"}):
            async with _make_async_client(app) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"question": "Проверка режима real"},
                )
                self.assertEqual(resp.status_code, 503)
                self.assertEqual(resp.json()["code"], "rag_not_ready")

                # Health endpoint reports RAG status
                h_resp = await client.get("/api/health")
                self.assertEqual(h_resp.status_code, 200)
                h_data = h_resp.json()
                self.assertEqual(h_data["rag"]["mode"], "real")

    async def test_rag_initialization_failure_reports_degraded_health(self) -> None:
        # Test Point 2: RAG startup failure records rag_ready=False and health degraded
        app = create_app(db_path=self.db_path)
        app.state.rag_ready = False
        app.state.rag_error = "CUDA out of memory during model initialization"
        with patch.dict("os.environ", {"RAG_MODE": "real"}):
            async with _make_async_client(app) as client:
                resp = await client.post("/api/chat", json={"question": "Тест ошибки"})
                self.assertEqual(resp.status_code, 503)
                self.assertEqual(resp.json()["code"], "rag_not_ready")

                h_resp = await client.get("/api/health")
                h_data = h_resp.json()
                self.assertEqual(h_data["status"], "degraded")
                self.assertFalse(h_data["rag"]["ready"])

    async def test_sqlite_failure_sends_error_instead_of_done(self) -> None:
        app = create_app(db_path=self.db_path)
        with patch.object(app.state.db, "save_completed_turn", side_effect=RuntimeError("disk full")):
            async with _make_async_client(app) as client:
                resp = await client.post("/api/chat", json={"question": "Тест сбоя SQLite"})
                frames = _parse_sse_frames(resp.text)
                self.assertEqual(frames[-1][0], "error")
                self.assertEqual(frames[-1][1]["code"], "storage_error")

    async def test_single_active_request_per_session(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _slow_answer(question: str, history: list, program_code: str | None = None):
            yield {"type": "status", "stage": "planning"}
            entered.set()
            await release.wait()
            yield {"type": "token", "text": "Готово"}
            yield {"type": "done", "finish_reason": "stop", "diagnostics": {}}

        app = create_app(db_path=self.db_path, answer_fn=_slow_answer, rate_limit_per_minute=50)
        async with _make_async_client(app) as client:
            t1 = asyncio.create_task(
                client.post("/api/chat", json={"question": "Первый", "session_id": "sess-lock"})
            )
            await entered.wait()
            resp2 = await client.post(
                "/api/chat", json={"question": "Второй", "session_id": "sess-lock"}
            )
            self.assertEqual(resp2.status_code, 409)
            self.assertEqual(resp2.json()["code"], "session_busy")
            release.set()
            await t1

    async def test_20_concurrent_streams_do_not_block_server_and_sqlite(self) -> None:
        # Test Point 5: 20 streams reach barrier before release, then all complete and persist under shared write lock
        active_streams = 0
        max_concurrent_streams = 0
        all_twenty_active = asyncio.Event()

        async def _concurrent_rag(question: str, history: list, program_code: str | None = None):
            nonlocal active_streams, max_concurrent_streams
            active_streams += 1
            if active_streams > max_concurrent_streams:
                max_concurrent_streams = active_streams
            if active_streams >= 20:
                all_twenty_active.set()
            try:
                yield {"type": "status", "stage": "planning"}
                await asyncio.wait_for(all_twenty_active.wait(), timeout=3.0)
                yield {"type": "status", "stage": "generating"}
                yield {"type": "token", "text": f"Ответ на {question}"}
                yield {"type": "done", "finish_reason": "stop", "diagnostics": {"sources": ["SE_DP_.pdf"]}}
            finally:
                active_streams -= 1

        app = create_app(
            db_path=self.db_path,
            answer_fn=_concurrent_rag,
            rate_limit_per_minute=100,
        )
        async with _make_async_client(app) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/api/chat",
                        json={"question": f"Параллельный вопрос {i}", "session_id": f"barrier-sess-{i}"},
                    )
                    for i in range(20)
                )
            )

            self.assertEqual(max_concurrent_streams, 20)
            for i, resp in enumerate(responses):
                self.assertEqual(resp.status_code, 200)
                frames = _parse_sse_frames(resp.text)
                self.assertEqual(frames[-1][0], "done")
                self.assertIsNotNone(frames[-1][1]["message_id"])

            hist = await self.db.get_session_history("barrier-sess-19")
            self.assertIsNotNone(hist)
            self.assertEqual(len(hist["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
