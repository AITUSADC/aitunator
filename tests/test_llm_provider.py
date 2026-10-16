import asyncio
import json
import unittest
import httpx
from backend.app.providers.llm.chat import ChatSettings, ChatCompletionsProvider, ProviderError


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, handler):
        provider = ChatCompletionsProvider(ChatSettings("https://test.invalid/v1", "test-key", "main", "fast"),
                                           transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(provider.close)
        return provider

    async def test_json_uses_fast_model_and_validates_finish(self):
        def handler(request):
            data = json.loads(request.content)
            self.assertEqual(data["model"], "fast")
            self.assertEqual(data["response_format"], {"type": "json_object"})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"sufficient":false}'}}]})
        provider = await self.make(handler)
        self.assertEqual(await provider.complete_json([]), {"sufficient": False})

    async def test_stream_success(self):
        body = 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        provider = await self.make(lambda request: httpx.Response(200, text=body))
        self.assertEqual([t async for t in provider.stream([])], ["Hello"])

    async def test_truncation_missing_terminal_and_malformed_data_fail(self):
        for body in ['data: [DONE]\n\n', 'data: {}\n\n', 'data: invalid\n\n',
                     'data: {"choices":[{"finish_reason":"length","delta":{}}]}\n\n']:
            provider = await self.make(lambda request, body=body: httpx.Response(200, text=body))
            with self.assertRaises(ProviderError):
                _ = [t async for t in provider.stream([])]

    async def test_http_errors_hide_body_and_classify_retries(self):
        for code, retryable in [(401, False), (429, True), (503, True)]:
            provider = await self.make(lambda request, code=code: httpx.Response(code, text="secret user content"))
            with self.assertRaises(ProviderError) as error:
                await provider.complete_json([])
            self.assertEqual(error.exception.retryable, retryable)
            self.assertNotIn("secret", str(error.exception))

    async def test_cancel_closes_http_response(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"unreachable"
            async def aclose(self):
                closed.set()
        provider = await self.make(lambda request: httpx.Response(200, stream=Body()))
        async def collect():
            return [t async for t in provider.stream([])]
        task = asyncio.create_task(collect())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
