"""Async chat-completions transport; no credentials, requests or clients at import."""
from dataclasses import dataclass, field
import json
import os
from typing import Protocol
from collections.abc import AsyncIterator

import httpx


class ProviderError(Exception):
    def __init__(self, code="llm_unavailable", retryable=True):
        super().__init__(code)  # Never include provider bodies/prompts/credentials.
        self.code, self.retryable = code, retryable


class ChatProvider(Protocol):
    async def complete_json(self, messages: list[dict]) -> dict: ...
    def stream(self, messages: list[dict]) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class ChatSettings:
    base_url: str
    api_key: str = field(repr=False)
    main_model: str
    fast_model: str
    timeout: float = 30.0
    temperature: float | None = 0.0  # None: reasoning models accept only their default

    @classmethod
    def from_env(cls, env=os.environ):
        provider = env.get("LLM_PROVIDER", "groq").lower()
        if provider == "groq":
            base = "https://api.groq.com/openai/v1"
            key = env.get("GROQ_API_KEY", "")
        elif provider == "compatible":
            base = env.get("LLM_BASE_URL", "")
            key = env.get("LLM_API_KEY", "")
        else:
            raise ValueError("LLM_PROVIDER must be groq or compatible")
        main = env.get("LLM_MAIN_MODEL") or env.get("LLM_MODEL", "")
        fast = env.get("LLM_FAST_MODEL") or main
        if not base or not key or not main:
            raise ValueError("Configure the LLM endpoint, API key and model before startup")
        temperature = env.get("LLM_TEMPERATURE", "0")
        return cls(base.rstrip("/"), key, main, fast, temperature=float(temperature) if temperature else None)


class ChatCompletionsProvider:
    def __init__(self, settings: ChatSettings, *, transport=None):
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.base_url + "/",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=httpx.Timeout(settings.timeout, connect=5.0),
            transport=transport, follow_redirects=False,
        )

    @property
    def _sampling(self):
        t = self.settings.temperature
        return {} if t is None else {"temperature": t}

    @staticmethod
    def _check(response):
        if response.is_error:
            raise ProviderError("llm_http_error", response.status_code == 429 or response.status_code >= 500)

    async def complete_json(self, messages):
        try:
            response = await self.client.post("chat/completions", json={
                "model": self.settings.fast_model, "messages": messages,
                "max_completion_tokens": 4096,  # reasoning models spend this budget on thinking too
                "response_format": {"type": "json_object"}, "stream": False, **self._sampling,
            })
            self._check(response)
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ProviderError("llm_incomplete")
            data = json.loads(choice["message"]["content"])
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except httpx.RequestError:
            raise ProviderError() from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("llm_invalid_response") from None

    async def stream(self, messages):
        finished = False
        try:
            async with self.client.stream("POST", "chat/completions", json={
                "model": self.settings.main_model, "messages": messages,
                "max_completion_tokens": 4096, "stream": True, **self._sampling,
            }) as response:
                self._check(response)
                data_lines = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        data = "\n".join(data_lines)
                        data_lines = []
                        if data == "[DONE]":
                            if not finished:
                                raise ProviderError("llm_incomplete")
                            return
                        event = json.loads(data)
                        if "error" in event:
                            raise ProviderError("llm_stream_error")
                        for choice in event.get("choices", []):
                            reason = choice.get("finish_reason")
                            if reason:
                                if reason != "stop":
                                    raise ProviderError("llm_incomplete")
                                finished = True
                            content = choice.get("delta", {}).get("content")
                            if content:
                                if not isinstance(content, str):
                                    raise ValueError()
                                yield content
                # Require both provider stop and the SSE terminal marker.
                raise ProviderError("llm_incomplete")
        except httpx.RequestError:
            raise ProviderError() from None
        except (ValueError, KeyError, TypeError):
            raise ProviderError("llm_invalid_response") from None

    async def close(self):
        await self.client.aclose()
