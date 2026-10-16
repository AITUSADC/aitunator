import asyncio
import json
import unittest
from unittest.mock import patch

from backend.app.rag.answering import AnswerService
from backend.app.providers.llm.chat import ProviderError

ROWS = ({"text": "Обучение длится три года.", "source": "program.md"},)
SUFFICIENT = {"sufficient": True, "evidence_ids": ["0"]}


class FakeProvider:
    def __init__(self, replies=(), tokens=("Три ", "года.")):
        self.replies = iter(replies)
        self.tokens = tokens
        self.calls = []
        self.stream_closed = False

    async def complete_json(self, messages):
        self.calls.append(messages)
        return next(self.replies)

    async def stream(self, messages):
        self.calls.append(messages)
        try:
            for token in self.tokens:
                if isinstance(token, Exception):
                    raise token
                yield token
        finally:
            self.stream_closed = True

    async def close(self):
        pass


class AnswerTests(unittest.IsolatedAsyncioTestCase):
    def service(self, replies=(), tokens=("Три года.",), rows=ROWS):
        self.retrievals = []
        async def retrieve(query, code):
            self.retrievals.append((query, code))
            return rows, {"total_ms": 1}, []
        self.provider = FakeProvider(replies, tokens)
        return AnswerService(self.provider, retrieve)

    async def collect(self, service, question="Срок обучения?", history=None, code=None):
        return [e async for e in service.answer(question, history or [], code)]

    async def test_grounded_stream_hides_sources_until_internal_done(self):
        events = await self.collect(self.service([SUFFICIENT]))
        self.assertEqual(events[-1]["finish_reason"], "stop")
        self.assertEqual(events[-1]["diagnostics"]["sources"], ["program.md"])
        self.assertEqual(len(self.retrievals), 1)
        self.assertTrue(self.provider.stream_closed)
        prompt_data = json.loads(self.provider.calls[-1][-1]["content"])
        self.assertNotIn("source", prompt_data["evidence"][0])

    async def test_guards_do_not_call_model_or_retrieval(self):
        for question in ["Мой ИИН 040512500123", "Гарантируешь поступление?", "Личный телефон декана", "My email a@example.com"]:
            events = await self.collect(self.service(), question)
            self.assertEqual(events[-1]["finish_reason"], "guardrail")
            self.assertEqual(self.provider.calls, [])
            self.assertEqual(self.retrievals, [])

    async def test_history_rewrite_filters_pii_and_caps_window(self):
        history = [{"role": "user", "content": f"Вопрос {i}"} for i in range(12)]
        history[-1] = {"role": "user", "content": "a@example.com"}
        events = await self.collect(self.service([{"query": "Срок обучения Software Engineering?"}, SUFFICIENT]),
                                    "А сколько учиться?", history)
        self.assertEqual(events[-1]["finish_reason"], "stop")
        sent = json.loads(self.provider.calls[0][-1]["content"])["history"]
        self.assertEqual(len(sent), 9)
        self.assertNotIn("example.com", json.dumps(sent))
        self.assertEqual(self.retrievals[0][0], "Срок обучения Software Engineering?")

    async def test_retry_is_bounded_and_original_question_preserved(self):
        insufficient = {"sufficient": False, "evidence_ids": [], "retry_query": "Срок бакалавриата"}
        events = await self.collect(self.service([insufficient, SUFFICIENT]))
        self.assertEqual(len(self.retrievals), 2)
        self.assertEqual(events[-1]["finish_reason"], "stop")
        data = json.loads(self.provider.calls[-1][-1]["content"])
        self.assertEqual(data["standalone_question"], "Срок обучения?")

    async def test_two_insufficient_results_refuse_without_generation(self):
        replies = [{"sufficient": False, "evidence_ids": [], "retry_query": q} for q in ("Другой запрос", "Третий запрос")]
        events = await self.collect(self.service(replies))
        self.assertEqual(len(self.retrievals), 2)
        self.assertEqual(events[-1]["finish_reason"], "no_data")
        self.assertFalse(self.provider.stream_closed)

    async def test_empty_corpus_refuses_without_llm(self):
        events = await self.collect(self.service(rows=()))
        self.assertEqual(events[-1]["finish_reason"], "no_data")
        self.assertEqual(self.provider.calls, [])

    async def test_invalid_judge_and_unrecognized_ids_never_generate(self):
        for reply in ({"sufficient": "true", "evidence_ids": ["0"]},
                      {"sufficient": True, "evidence_ids": ["999"]}):
            events = await self.collect(self.service([reply]))
            self.assertNotEqual(events[-1].get("finish_reason"), "stop")
            self.assertFalse(self.provider.stream_closed)

    async def test_provider_failure_after_token_is_error_not_done(self):
        events = await self.collect(self.service([SUFFICIENT], tokens=("Часть", ProviderError("llm_incomplete"))))
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("done", [e["type"] for e in events])
        self.assertTrue(self.provider.stream_closed)

    async def test_cancellation_closes_provider_stream(self):
        entered = asyncio.Event()
        closed = asyncio.Event()
        service = self.service([SUFFICIENT])
        async def stream(_):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()
        self.provider.stream = stream
        task = asyncio.create_task(self.collect(service))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())

    async def test_runtime_not_ready_and_lifecycle(self):
        from backend.app.rag import runtime
        await runtime.shutdown()
        events = [e async for e in runtime.answer("Вопрос", [])]
        self.assertEqual(events[-1]["code"], "rag_not_ready")
        with patch.object(runtime.ChatSettings, "from_env", return_value=object()), \
             patch.object(runtime, "load_retriever", side_effect=ValueError("bad vectors")):
            with self.assertRaises(ValueError):
                await runtime.startup()
        self.assertFalse(runtime.is_ready())

    async def test_startup_is_idempotent_and_shutdown_closes_provider(self):
        from backend.app.rag import runtime
        from unittest.mock import AsyncMock, Mock
        await runtime.shutdown()
        provider = Mock(close=AsyncMock())
        retriever = Mock(retrieve=AsyncMock())
        with patch.object(runtime.ChatSettings, "from_env", return_value=object()), \
             patch.object(runtime, "load_retriever", return_value=retriever) as load, \
             patch.object(runtime, "ChatCompletionsProvider", return_value=provider):
            await runtime.startup()
            await runtime.startup()
            self.assertTrue(runtime.is_ready())
            load.assert_called_once()
            await runtime.shutdown()
            await runtime.shutdown()
        provider.close.assert_awaited_once()
        self.assertFalse(runtime.is_ready())

    async def test_pii_evidence_is_excluded_without_losing_source_mapping(self):
        rows = ({"text": "Email: a@example.com", "source": "private.md"}, ROWS[0])
        events = await self.collect(self.service([{"sufficient": True, "evidence_ids": ["1"]}], rows=rows))
        self.assertEqual(events[-1]["diagnostics"]["sources"], ["program.md"])
        self.assertNotIn("example.com", json.dumps(self.provider.calls))


if __name__ == "__main__":
    unittest.main()
