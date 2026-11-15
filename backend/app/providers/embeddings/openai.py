"""OpenAI-compatible /embeddings with the SentenceTransformer.encode() surface used by the pipeline."""
import os

import httpx
import numpy as np

API_PREFIX = "openai/"


class OpenAIEmbedder:
    prompts: dict = {}

    def __init__(self, model: str, client: httpx.Client, dimensions: int = 1024, batch_size: int = 64):
        self.model, self.client, self.dimensions, self.batch_size = model, client, dimensions, batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post("embeddings", json={"model": self.model, "input": texts,
                                                        "dimensions": self.dimensions})
        response.raise_for_status()
        return [item["embedding"] for item in sorted(response.json()["data"], key=lambda item: item["index"])]

    def encode(self, texts, normalize_embeddings=True, **_options):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        vectors = np.asarray([vector for i in range(0, len(texts), self.batch_size)
                              for vector in self._embed(texts[i:i + self.batch_size])], dtype=np.float32)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors[0] if single else vectors


def load_api_embedder(name: str, env=os.environ) -> OpenAIEmbedder | None:
    """`openai/<model>` selects the API; other names stay local Sentence Transformers models."""
    if not name.startswith(API_PREFIX):
        return None
    key = env.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required for openai/ embedding models")
    base = (env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/"
    client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=60)
    return OpenAIEmbedder(name.removeprefix(API_PREFIX), client)
