"""Synchronous retrieval plus lazy async entry points for the API contract."""
from collections import defaultdict
from time import perf_counter
from backend.app.rag.retriever import retrieve_candidates, rank_candidates
from backend.app.rag.planner import program_queries
from backend.app.rag.structured import balanced_results, prefer_sections


def search(query, *, corpus, encode, fanout=True, candidates=8,
           section_selection=True, program_code=None, doc_types=(), language=None):
    start = perf_counter()
    queries = program_queries(query) if fanout and not program_code else []
    queries = queries or [(program_code, query)]
    groups, stages = [], defaultdict(float)
    for code, subquery in queries:
        batch = retrieve_candidates(subquery, k=candidates, mode="hybrid",
            corpus=corpus.rows, bm25=corpus.bm25,
            encode_query=encode,
            dense_search=corpus.dense_search, program_code=code,
            doc_types=tuple(doc_types or ()), language=language)
        results, ranking = rank_candidates(batch, top_k=candidates if section_selection else 5)
        if section_selection:
            tick = perf_counter()
            results = prefer_sections(results, subquery)[:5]
            stages["section_selection_ms"] += (perf_counter() - tick) * 1000
        groups.append(results)
        for key, value in {**batch.timings, **ranking}.items():
            stages[key] += value
    results = balanced_results(groups)
    stages["total_ms"] = (perf_counter() - start) * 1000
    return results, dict(stages), queries


async def startup():
    from backend.app.rag.runtime import startup as initialize
    await initialize()


async def shutdown():
    from backend.app.rag.runtime import shutdown as close
    await close()


def is_ready():
    from backend.app.rag.runtime import is_ready as ready
    return ready()


async def answer(question, history, program_code=None):
    from contextlib import aclosing
    from backend.app.rag.runtime import answer as generate
    async with aclosing(generate(question, history, program_code)) as events:
        async for event in events:
            yield event
