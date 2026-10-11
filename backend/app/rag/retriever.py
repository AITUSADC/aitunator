"""Candidate retrieval and ranking, independent of model/server initialization.

Scores are diagnostics, not calibrated answer probabilities. Dense callbacks return
{id, payload, score}; both the Qdrant adapter and offline benchmark use this pipeline.
"""
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from backend.app.rag.retrieval_common import (
    _GENERAL_MOBILITY_RE, _apply_query_program_scope, _expand_query_aliases,
    _metadata_boost, _normalize_scope, _payload_matches_scope, _tokenize,
)


@dataclass
class CandidateBatch:
    query: str
    scope: dict
    candidates: tuple[dict, ...]
    timings: dict[str, float]  # milliseconds; no model loading or LRU cache hits


def retrieve_candidates(
    query: str, *, k: int, mode: str, corpus: list[dict], bm25,
    encode_query: Callable | None = None, dense_search: Callable | None = None,
    source_filter=None, program_code=None, doc_types=(), language=None,
) -> CandidateBatch:
    if mode not in {"bm25", "dense", "hybrid"} or k < 1:
        raise ValueError("Expected bm25/dense/hybrid and positive k")
    start = perf_counter()
    scope = _apply_query_program_scope(
        _normalize_scope(source_filter, program_code, doc_types, language), query,
    )
    if _GENERAL_MOBILITY_RE.search(query):
        scope = {**scope, "source": None, "program_code": None, "program_slug": None}
    search_query = _expand_query_aliases(query, scope)
    timings = {"embedding_ms": 0.0, "dense_ms": 0.0, "bm25_ms": 0.0}
    lexical, dense = [], []
    if mode in {"bm25", "hybrid"} and bm25 is not None:
        tick = perf_counter()
        scores = bm25.get_scores(_tokenize(search_query))
        # Filter BEFORE top-k, otherwise valid scoped hits can be silently lost.
        eligible = [i for i, row in enumerate(corpus)
                    if scores[i] > 0 and _payload_matches_scope(row["payload"], scope)]
        indices = sorted(eligible, key=lambda i: (-float(scores[i]), str(corpus[i]["id"])))[:k]
        lexical = [{**corpus[i], "score": float(scores[i])} for i in indices]
        timings["bm25_ms"] = (perf_counter() - tick) * 1000
    if mode in {"dense", "hybrid"}:
        if encode_query is None or dense_search is None:
            raise ValueError("Dense retrieval needs an encoder and dense-search adapter")
        tick = perf_counter()
        vector = encode_query(search_query)
        timings["embedding_ms"] = (perf_counter() - tick) * 1000
        tick = perf_counter()
        dense = [row for row in dense_search(vector, scope, k)
                 if _payload_matches_scope(row["payload"], scope)]
        timings["dense_ms"] = (perf_counter() - tick) * 1000

    tick = perf_counter()
    by_id = {}
    for method, rows in (("dense", dense), ("bm25", lexical)):
        for rank, row in enumerate(rows, 1):
            key = str(row["id"])
            if key not in by_id:
                by_id[key] = {**row["payload"], "_retrieval": {"id": key, "rrf_score": 0.0}}
            scores = by_id[key]["_retrieval"]
            scores[f"{method}_score"] = float(row["score"])
            scores[f"{method}_rank"] = rank
            scores["rrf_score"] += 1 / (60 + rank)
    candidates = sorted(by_id.values(), key=lambda row: -row["_retrieval"]["rrf_score"])[:k]
    timings["fusion_ms"] = (perf_counter() - tick) * 1000
    timings["retrieval_ms"] = (perf_counter() - start) * 1000
    return CandidateBatch(search_query, scope, tuple(candidates), timings)


def rank_candidates(batch: CandidateBatch, *, top_k: int, predict=None,
                    metadata_boost: bool = False, original_query: str | None = None):
    """Rank existing candidates; never re-embed or retrieve again."""
    start = perf_counter()
    results = [{**row, "_retrieval": dict(row["_retrieval"])} for row in batch.candidates]
    if predict is not None and results:
        values = predict([[batch.query, row.get("text", "")] for row in results])
        if len(values) != len(results):
            raise ValueError("Reranker returned an incorrect number of scores")
        for row, value in zip(results, values):
            row["_retrieval"]["rerank_score"] = float(value)
        def score(row):
            boost = _metadata_boost(original_query or batch.query, row, batch.scope) if metadata_boost else 0.0
            return row["_retrieval"]["rerank_score"] + boost
        results.sort(key=score, reverse=True)
    elif metadata_boost:
        # Preserve the old application's metadata heuristic; benchmarks default
        # to pure retrieval ordering, so they measure BM25/dense/RRF themselves.
        results = [row for _, row in sorted(enumerate(results), key=lambda item:
            _metadata_boost(original_query or batch.query, item[1], batch.scope)
            + (len(results) - item[0]) / 1000, reverse=True)]
    elapsed = (perf_counter() - start) * 1000
    return tuple(results[:top_k]), {"rerank_ms": elapsed if predict else 0.0, "ranking_ms": elapsed}


def retrieval_features(batch: CandidateBatch) -> dict:
    """Raw routing features for later calibration; deliberately no threshold."""
    results = batch.candidates
    if not results:
        return {"candidate_count": 0, "bm25_top_score": 0.0, "bm25_gap": 0.0,
                "term_coverage": 0.0, "agreement_at_5": 0.0}
    lexical = sorted((row for row in results if "bm25_rank" in row["_retrieval"]),
                     key=lambda row: row["_retrieval"]["bm25_rank"])
    first = lexical[0]["_retrieval"]["bm25_score"] if lexical else 0.0
    second = lexical[1]["_retrieval"]["bm25_score"] if len(lexical) > 1 else 0.0
    query_terms = set(_tokenize(batch.query))
    text_terms = set(_tokenize(lexical[0].get("text", ""))) if lexical else set()
    bm_ids = {r["_retrieval"]["id"] for r in results if r["_retrieval"].get("bm25_rank", 999) <= 5}
    dense_ids = {r["_retrieval"]["id"] for r in results if r["_retrieval"].get("dense_rank", 999) <= 5}
    return {"candidate_count": len(results), "bm25_top_score": first,
            "bm25_gap": (first - second) / first if first > 0 else 0.0,
            "term_coverage": len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0,
            "agreement_at_5": len(bm_ids & dense_ids) / len(bm_ids | dense_ids) if bm_ids | dense_ids else 0.0}
