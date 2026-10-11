"""Compare retrievers on the same saved corpus without a running Qdrant or LLM.

Dense search is exact cosine over cached document vectors, NOT Qdrant latency.
Every timed mode computes its own query embedding; no shared query-result cache.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
from statistics import mean
from time import perf_counter

import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.rag.retriever import retrieve_candidates, rank_candidates, retrieval_features
from backend.app.providers.embeddings.sentence_transformer import encode_query, encode_documents
from backend.app.rag.retrieval_common import _payload_matches_scope, _tokenize, _ru_stops, _en_stops

ROOT = Path(__file__).resolve().parents[1]
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def package_versions():
    versions = {}
    for name in ("numpy", "rank-bm25", "nltk", "torch", "sentence-transformers", "transformers"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def model_location(name, cache, download):
    """Resolve the actual snapshot path to avoid Hub metadata calls offline."""
    if Path(name).is_dir() or download:
        return name
    repository = cache / ("models--" + name.replace("/", "--"))
    ref = repository / "refs/main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = (repository / "snapshots" / revision).resolve()
        if snapshot.is_relative_to(repository.resolve()) and snapshot.is_dir():
            return str(snapshot)
    raise FileNotFoundError(f"No local model snapshot for {name}; use --download-models first")


def read_cases(path):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Dataset must be nonempty and case IDs unique")
    for case in cases:
        if case["answerable"] and not case.get("evidence_groups"):
            raise ValueError(f"Missing evidence labels: {case['id']}")
    return cases


def source_key(value):
    return str(value).replace("\\", "/").removeprefix("./")


def metrics(case, results):
    if not case["answerable"]:
        # A retriever returns candidates even for unanswerable questions. That is
        # not an incorrect answer: refusal must be measured in the answer layer.
        return {"hit_at_5": None, "all_evidence_at_5": None, "mrr_at_5": None,
                "section_evidence_at_5": None}
    groups = [{source_key(v) for v in group} for group in case["evidence_groups"]]
    sources = [source_key(row.get("source", "")) for row in results[:5]]
    ranks = [i for i, source in enumerate(sources, 1) if any(source in group for group in groups)]
    section_groups = case.get("section_evidence_groups", [])
    sections_found = {(source_key(row.get("source", "")), row.get("section")) for row in results[:5]}
    section_hit = int(all(any((source_key(target["source"]), target["section"]) in sections_found
                             for target in group) for group in section_groups)) if section_groups else None
    return {"hit_at_5": int(bool(ranks)), "section_evidence_at_5": section_hit,
            "all_evidence_at_5": int(all(any(source in group for source in sources) for group in groups)),
            "mrr_at_5": 1 / min(ranks) if ranks else 0.0}


def summarize(rows):
    positives = [row for row in rows if row["metrics"]["hit_at_5"] is not None]
    durations = [row["timings"]["total_ms"] for row in rows]
    annotated = [row for row in rows if row["metrics"]["section_evidence_at_5"] is not None]
    return {"runs": len(rows), "answerable_runs": len(positives),
            "section_annotated_runs": len(annotated),
            "section_evidence_at_5": mean(row["metrics"]["section_evidence_at_5"] for row in annotated) if annotated else None,
            **{metric: mean(row["metrics"][metric] for row in positives) if positives else None
               for metric in ("hit_at_5", "all_evidence_at_5", "mrr_at_5")},
            "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95)),
            "stage_mean_ms": {key: mean(row["timings"].get(key, 0.0) for row in rows)
                              for key in ("bm25_ms", "embedding_ms", "dense_ms", "fusion_ms", "rerank_ms")}}


class OfflineCorpus:
    def __init__(self, path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows or len({str(r['id']) for r in self.rows}) != len(self.rows):
            raise ValueError("Corpus is empty or contains duplicate IDs")
        texts = [" ".join(str(row["payload"].get(key) or "") for key in
                 ("program_name", "program_code", "section", "text")) for row in self.rows]
        self.bm25 = BM25Okapi([_tokenize(text) for text in texts])
        self.vectors = np.asarray([row["vector"] for row in self.rows], dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if not np.isfinite(self.vectors).all() or np.any(norms == 0):
            raise ValueError("Corpus contains invalid dense vectors")
        self.vectors /= norms

    def dense_search(self, vector, scope, k):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all() or norm == 0:
            raise ValueError("Query vector is invalid or incompatible with cached corpus")
        eligible = [i for i, row in enumerate(self.rows) if _payload_matches_scope(row["payload"], scope)]
        if not eligible:
            return []
        scores = self.vectors[eligible] @ (vector / norm)
        top = sorted(range(len(eligible)), key=lambda j: (-float(scores[j]), str(self.rows[eligible[j]]["id"])))[:k]
        return [{**self.rows[eligible[j]], "score": float(scores[j])} for j in top]


def run(args):
    # Transformers can otherwise start a background remote conversion task for
    # legacy .bin weights. Benchmark inference needs neither conversion nor it.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
    if not args.download_models:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps({"complete": False, "status": "running"}), encoding="utf-8")
    cases = read_cases(args.dataset)
    corpus = OfflineCorpus(args.corpus)
    known_sources = {source_key(row["payload"]["source"]) for row in corpus.rows}
    for case in cases:
        for group in case.get("evidence_groups", []):
            if not set(map(source_key, group)) & known_sources:
                raise ValueError(f"Evidence absent from corpus: {case['id']}: {group}")
        known_sections = {(source_key(row["payload"]["source"]), row["payload"].get("section")) for row in corpus.rows}
        for group in case.get("section_evidence_groups", []):
            if not any((source_key(item["source"]), item["section"]) in known_sections for item in group):
                raise ValueError(f"Section evidence absent from corpus: {case['id']}: {group}")
    if args.limit:
        cases = cases[:args.limit]
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    if not cases:
        raise ValueError("No cases selected")
    encoder = reranker = None
    unavailable = {}
    modes = list(args.modes)
    loading_ms = {}
    if any(mode != "bm25" for mode in modes):
        try:
            print("Loading encoder...", flush=True)
            import torch
            torch.set_num_threads(args.threads)
            from sentence_transformers import SentenceTransformer
            tick = perf_counter()
            encoder = SentenceTransformer(model_location(args.embedding_model, args.model_cache, args.download_models), device=args.device,
                cache_folder=str(args.model_cache), local_files_only=not args.download_models)
            loading_ms["encoder"] = (perf_counter() - tick) * 1000
            print(f"Encoder loaded in {loading_ms['encoder'] / 1000:.1f}s", flush=True)
        except (ImportError, OSError, RuntimeError) as exc:
            unavailable.update({mode: f"Encoder unavailable: {type(exc).__name__}: {exc}"
                                for mode in modes if mode != "bm25"})
            modes = [mode for mode in modes if mode == "bm25"]
    if "hybrid_rerank" in modes:
        try:
            print("Loading reranker...", flush=True)
            from sentence_transformers import CrossEncoder
            tick = perf_counter()
            reranker = CrossEncoder(model_location(args.reranker_model, args.model_cache, args.download_models), device=args.device,
                max_length=args.rerank_max_length, cache_folder=str(args.model_cache),
                local_files_only=not args.download_models)
            loading_ms["reranker"] = (perf_counter() - tick) * 1000
            print(f"Reranker loaded in {loading_ms['reranker'] / 1000:.1f}s", flush=True)
        except (ImportError, OSError, RuntimeError) as exc:
            unavailable["hybrid_rerank"] = f"Reranker unavailable: {type(exc).__name__}: {exc}"
            modes.remove("hybrid_rerank")
    if not modes:
        raise RuntimeError(f"No requested modes available: {unavailable}")

    vector_check = None
    if encoder is not None:
        from backend.app.ingest.extract import _embedding_prefix
        sample_ids = sorted({0, len(corpus.rows) // 2, len(corpus.rows) - 1})
        selected = list(range(len(corpus.rows))) if args.reembed_corpus else sample_ids
        texts = [f"{_embedding_prefix(corpus.rows[i]['payload'])} {corpus.rows[i]['payload']['text']}" for i in selected]
        tick = perf_counter()
        vectors = encode_documents(encoder, texts, batch_size=8, show_progress_bar=False)
        loading_ms["corpus_encoding_or_verification"] = (perf_counter() - tick) * 1000
        if args.reembed_corpus:
            corpus.vectors = np.asarray(vectors, dtype=np.float32)
            vector_check = {"method": "reencoded_in_memory", "original_cache_modified": False}
        else:
            if np.asarray(vectors).shape[1] != corpus.vectors.shape[1]:
                raise ValueError("Embedding dimensions differ; run with --reembed-corpus")
            similarities = [float(np.dot(v, corpus.vectors[i])) for i, v in zip(selected, vectors)]
            vector_check = {"sample_indices": selected, "cosines": similarities, "minimum_required": 0.995}
            if min(similarities) < 0.995:
                raise ValueError(f"Cached vectors do not match encoder/prefix on sampled documents: {similarities}. "
                                 "Run with --reembed-corpus; it preserves the original cache.")

    def encode(text):
        # BGE-M3 has no mandatory query prefix. Preserve named prompts when
        # present, but don't require the legacy project's undefined 'query'.
        return encode_query(encoder, text, show_progress_bar=False)

    def predict(pairs):
        return reranker.predict(pairs, batch_size=args.rerank_batch_size, show_progress_bar=False)

    def measure(case, mode):
        tick = perf_counter()
        batch = retrieve_candidates(
            case.get("standalone_question", case["question"]), k=args.candidates,
            mode="hybrid" if mode == "hybrid_rerank" else mode,
            corpus=corpus.rows, bm25=corpus.bm25, encode_query=encode,
            dense_search=corpus.dense_search, program_code=case.get("program_code"),
            doc_types=tuple(case.get("doc_types") or ()), language=case.get("language"),
        )
        results, ranking = rank_candidates(batch, top_k=5,
            predict=predict if mode == "hybrid_rerank" else None,
            metadata_boost=args.metadata_boost,
            original_query=case.get("standalone_question", case["question"]))
        elapsed = (perf_counter() - tick) * 1000
        return {"case_id": case["id"], "mode": mode, "category": case["category"],
                "split": case["split"], "answerable": case["answerable"],
                "question": case["question"], "metrics": metrics(case, results),
                "features": retrieval_features(batch),
                "timings": {**batch.timings, **ranking, "total_ms": elapsed},
                "results": [{"source": row.get("source"), "section": row.get("section"),
                             **row["_retrieval"]} for row in results]}

    # Warm each model/path; initialization and first-call costs are reported
    # separately. No query or candidate caching in this benchmark.
    warmup = {}
    for mode in modes:
        print(f"Warming {mode}...", flush=True)
        warmup[mode] = measure(cases[0], mode)["timings"]["total_ms"]
        print(f"Warmup {mode}: {warmup[mode]:.0f}ms", flush=True)
    rows = []
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runs.jsonl").open("w", encoding="utf-8") as out:
        for repeat in range(args.repeats):
            order = list(cases)
            rng.shuffle(order)
            for i, case in enumerate(order, 1):
                mode_order = list(modes)
                rng.shuffle(mode_order)
                for mode in mode_order:
                    row = {**measure(case, mode), "repeat": repeat}
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                print(f"[{repeat + 1}/{args.repeats}] {i}/{len(cases)} {case['id']}", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"]].append(row)
    report = {
        "complete": not unavailable, "status": "partial" if unavailable else "completed", "unavailable_modes": unavailable,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "corpus_chunks": len(corpus.rows), "cases": len(cases),
        "categories": dict(Counter(c["category"] for c in cases)),
        "settings": {**vars(args), "platform": platform.platform(),
                     "stopwords_ru": len(_ru_stops), "stopwords_en": len(_en_stops)},
        "model_loading_ms": loading_ms, "warmup_ms": warmup,
        "corpus_vector_check": vector_check,
        "package_versions": package_versions(),
        "summary": {mode: summarize(group) for mode, group in groups.items()},
        "by_category": {mode: {category: summarize([r for r in group if r["category"] == category])
                               for category in sorted({r['category'] for r in group})}
                        for mode, group in groups.items()},
        "by_split": {mode: {split: summarize([r for r in group if r["split"] == split])
                            for split in sorted({r['split'] for r in group})}
                     for mode, group in groups.items()},
        "notes": ["Exact local cosine over legacy cached vectors; no Qdrant latency.",
                  "Legacy cache has no model manifest; sampled vector compatibility checked, or corpus reencoded in memory.",
                  "Source-level metrics do not establish factual correctness or answerability.",
                  "Unanswerable cases excluded from hit/MRR; raw scores retained for calibration.",
                  "Follow-ups use manually resolved standalone queries; planner quality is not measured.",
                  "No routing threshold fitted or activated by this benchmark."],
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if unavailable:
        print("INCOMPLETE: " + ", ".join(unavailable), flush=True)
    return 0 if not unavailable or args.allow_partial else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/golden.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "runtime/points_cache.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/retrieval-benchmark")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["all", "calibration", "holdout"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "tmp/models")
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--metadata-boost", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--reembed-corpus", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if (args.repeats < 1 or args.candidates < 5 or args.threads < 1
            or args.rerank_batch_size < 1 or args.rerank_max_length < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error("Need repeats >= 1, candidates >= 5 and limit >= 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
