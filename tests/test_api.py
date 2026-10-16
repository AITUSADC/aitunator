"""Comprehensive unit and integration tests for SQLite, PII filtering, SSE chat, and API endpoints."""
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
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, \x27user\x27, \x27Привет\x27, ?);",
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

    async def test_pii_detector_covers_iin_phone_email_and_cards(self) -> None:
        self.assertTrue(contains_pii("Мой ИИН 040512500123, проверьте списки"))
        self.assertTrue(contains_pii("Позвоните мне на +7 701 234 56 78"))
        self.assertTrue(contains_pii("Моя почта student123@gmail.com"))
        self.assertTrue(contains_pii("Карта 4400 4301 1234 5678"))
        self.assertFalse(contains_pii("Какие проходные баллы на программу 6B06102 Software Engineering?"))

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
            self.assertNotIn("diagnostics", done_data)

            hist_resp = await client.get(f"/api/sessions/{session_id}")
            self.assertEqual(hist_resp.status_code, 200)
            hist_json = hist_resp.json()
            self.assertEqual(len(hist_json["messages"]), 2)
            self.assertEqual(hist_json["messages"][0]["role"], "user")
            self.assertEqual(hist_json["messages"][1]["id"], message_id)
            async with _connect(self.db_path) as conn:
                cur = await conn.execute("SELECT sources_json FROM messages WHERE id = ?;", (message_id,))
                row = await cur.fetchone()
                self.assertIsNotNone(row)
                self.assertIn("SE_DP_.pdf", row["sources_json"])

            fb_resp = await client.post(
                "/api/feedback",
                json={"message_id": message_id, "rating": 1, "comment": "Полезный ответ!"},
            )
            self.assertEqual(fb_resp.status_code, 200)

            fb_pii_resp = await client.post(
                "/api/feedback",
                json={
                    "message_id": message_id,
                    "rating": -1,
                    "comment": "Мой телефон +7 707 111 22 33, перезвоните",
                },
            )
            self.assertEqual(fb_pii_resp.status_code, 422)
            self.assertEqual(fb_pii_resp.json()["code"], "pii_detected")

    async def test_pii_and_guardrail_turns_are_not_saved_to_sqlite(self) -> None:
        async def _leaky_answer(question: str, history: list, program_code: str | None = None):
            yield {"type": "status", "stage": "generating"}
            yield {"type": "token", "text": "Личный телефон сотрудника: +7 701 999 88 77"}
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

        app_default = create_app(db_path=self.db_path)
        async with _make_async_client(app_default) as client:
            resp2 = await client.post(
                "/api/chat",
                json={"question": "Мой ИИН 050101500999, поступлю ли я на грант?", "session_id": "pii-session-2"},
            )
            frames2 = _parse_sse_frames(resp2.text)
            done_data2 = frames2[-1][1]
            self.assertEqual(done_data2["finish_reason"], "guardrail")
            self.assertIsNone(done_data2["message_id"])
            self.assertIsNone(await self.db.get_session_history("pii-session-2"))

    async def test_sqlite_failure_sends_error_instead_of_done(self) -> None:
        app = create_app(db_path=self.db_path)
        with patch.object(app.state.db, "save_completed_turn", side_effect=RuntimeError("disk full")):
            async with _make_async_client(app) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"question": "Сколько длится обучение в AITU?"},
                )
                frames = _parse_sse_frames(resp.text)
                self.assertEqual(frames[-1][0], "error")
                self.assertEqual(frames[-1][1]["code"], "storage_error")
                self.assertNotIn("done", [name for name, _ in frames])

    async def test_single_active_request_per_session_and_parallel_sessions(self) -> None:
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
            task1 = asyncio.create_task(
                client.post("/api/chat", json={"question": "Первый", "session_id": "same-session"})
            )
            await entered.wait()

            resp_duplicate = await client.post(
                "/api/chat", json={"question": "Второй параллельный", "session_id": "same-session"}
            )
            self.assertEqual(resp_duplicate.status_code, 409)
            self.assertEqual(resp_duplicate.json()["code"], "session_busy")

            release.set()
            resp_other = await client.post(
                "/api/chat", json={"question": "Другая сессия", "session_id": "other-session"}
            )
            self.assertEqual(resp_other.status_code, 200)
            resp1 = await task1
            self.assertEqual(resp1.status_code, 200)

    async def test_rate_limit_and_trusted_proxy_headers(self) -> None:
        app = create_app(db_path=self.db_path, rate_limit_per_minute=2)
        async with _make_async_client(app) as client:
            r1 = await client.post(
                "/api/chat",
                json={"question": "Вопрос 1"},
                headers={"X-Forwarded-For": "1.1.1.1"},
            )
            r2 = await client.post(
                "/api/chat",
                json={"question": "Вопрос 2"},
                headers={"X-Forwarded-For": "2.2.2.2"},
            )
            r3 = await client.post(
                "/api/chat",
                json={"question": "Вопрос 3"},
                headers={"X-Forwarded-For": "3.3.3.3"},
            )
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r3.status_code, 429)
            self.assertEqual(r3.json()["code"], "rate_limit")

    async def test_programs_and_health_endpoints(self) -> None:
        app = create_app(db_path=self.db_path)
        async with _make_async_client(app) as client:
            prog_resp = await client.get("/api/programs")
            self.assertEqual(prog_resp.status_code, 200)
            prog_data = prog_resp.json()
            self.assertEqual(prog_data["count"], 16)
            codes = {p["program_code"] for p in prog_data["programs"]}
            self.assertIn("6B06102", codes)

            health_resp = await client.get("/api/health")
            self.assertEqual(health_resp.status_code, 200)
            health_data = health_resp.json()
            self.assertEqual(health_data["status"], "ok")
            self.assertTrue(health_data["sqlite"]["available"])
            self.assertEqual(health_data["sqlite"]["journal_mode"], "wal")


if __name__ == "__main__":
    unittest.main()
