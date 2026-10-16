"""Explicit local runtime. Qdrant service wiring remains a separate adapter."""
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import threading

from backend.app.providers.llm.chat import ChatSettings, ChatCompletionsProvider
from backend.app.rag.answering import AnswerService


@dataclass
class LocalRetriever:
    corpus: object
    model: object
    candidates: int
    lock: object

    def search(self, query, program_code):
        from backend.app.rag.pipeline import search
        from backend.app.providers.embeddings.sentence_transformer import encode_query
        # Lock stays in the worker: cancellation cannot release it while encoding
        # is still running. No event-loop blocking or parallel access to the model.
        with self.lock:
            return search(query, corpus=self.corpus,
                          encode=lambda text: encode_query(self.model, text, show_progress_bar=False),
                          candidates=self.candidates, program_code=program_code)

    async def retrieve(self, query, program_code):
        return await asyncio.to_thread(self.search, query, program_code)


def load_retriever():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from backend.app.rag.corpus import OfflineCorpus
    from backend.app.providers.embeddings.sentence_transformer import encode_documents
    from backend.app.ingest.extract import _embedding_prefix

    backend = os.getenv("RAG_SEARCH_BACKEND", "local")
    if backend != "local":
        raise ValueError("This runtime supports RAG_SEARCH_BACKEND=local only")
    corpus = OfflineCorpus(Path(os.getenv("RAG_CORPUS_PATH", "runtime/points_cache.jsonl")))
    name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    cache = Path(os.getenv("MODEL_CACHE_DIR", "tmp/models"))
    download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "0") == "1"
    if not Path(name).is_dir() and not download:
        repository = cache / ("models--" + name.replace("/", "--"))
        ref = repository / "refs/main"
        if not ref.is_file():
            raise ValueError("Embedding model missing; configure MODEL_CACHE_DIR or allow download")
        snapshot = (repository / "snapshots" / ref.read_text().strip()).resolve()
        if not snapshot.is_relative_to(repository.resolve()) or not snapshot.is_dir():
            raise ValueError("Invalid model snapshot")
        name = str(snapshot)
    threads = int(os.getenv("RAG_CPU_THREADS", "4"))
    candidates = int(os.getenv("RAG_CANDIDATES", "8"))
    if threads < 1 or not 5 <= candidates <= 100:
        raise ValueError("Invalid retrieval thread/candidate limits")
    torch.set_num_threads(threads)
    model = SentenceTransformer(name, device="cpu", cache_folder=str(cache), local_files_only=not download)
    indices = sorted({0, len(corpus.rows) // 2, len(corpus.rows) - 1})
    texts = [f"{_embedding_prefix(corpus.rows[i]['payload'])} {corpus.rows[i]['payload']['text']}" for i in indices]
    vectors = encode_documents(model, texts, show_progress_bar=False)
    if vectors.shape[1] != corpus.vectors.shape[1] or any(
            float(np.dot(v, corpus.vectors[i])) < .995 for i, v in zip(indices, vectors)):
        raise ValueError("Cached vectors do not match the configured embedding model; reindex first")
    return LocalRetriever(corpus, model, candidates, threading.Lock())


_service = None
_lifecycle_lock = asyncio.Lock()


async def startup():
    global _service
    async with _lifecycle_lock:
        if _service is not None:
            return
        settings = ChatSettings.from_env()  # Fail before loading models if misconfigured.
        retriever = await asyncio.to_thread(load_retriever)
        provider = ChatCompletionsProvider(settings)
        _service = AnswerService(provider, retriever.retrieve)


async def shutdown():
    global _service
    async with _lifecycle_lock:
        service, _service = _service, None
        if service is not None:
            await service.provider.close()


def is_ready():
    return _service is not None


async def answer(question, history, program_code=None):
    service = _service
    if service is None:
        yield {"type": "error", "code": "rag_not_ready", "message": "Поиск и модель ещё не загружены.", "retryable": False}
        return
    from contextlib import aclosing
    async with aclosing(service.answer(question, history, program_code)) as events:
        async for event in events:
            yield event
