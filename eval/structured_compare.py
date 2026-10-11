"""Four-way offline ablation: legacy/cards x single search/program fan-out.

No production index changes, LLM calls, confidence thresholds or label-based routing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

from eval.run import (ROOT, OfflineCorpus, metrics, model_location,
                                         package_versions, read_cases, summarize)
from backend.app.providers.embeddings.sentence_transformer import encode_documents, encode_query
from backend.app.rag.pipeline import search
from backend.app.rag.structured import make_card, source_urls


def measure(case, corpus, encoder, fanout, candidates, section_selection=False):
    query = case.get("standalone_question", case["question"])
    results, stages, queries = search(query, corpus=corpus,
        encode=lambda text: encode_query(encoder, text, show_progress_bar=False),
        fanout=fanout, candidates=candidates, section_selection=section_selection,
        program_code=case.get("program_code"), doc_types=case.get("doc_types"),
        language=case.get("language"))
    return {"case_id": case["id"], "category": case["category"], "split": case["split"],
            "question": case["question"], "answerable": case["answerable"],
            "subqueries": queries, "metrics": metrics(case, results), "timings": dict(stages),
            "results": [{"source": r.get("source"), "section": r.get("section"),
                         **r["_retrieval"]} for r in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/structured-experiment")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.candidates < 5 or args.threads < 1:
        parser.error("Need positive repeats/threads and candidates >= 5")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      DISABLE_SAFETENSORS_CONVERSION="true")
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary_path.write_text('{"complete": false, "status": "running"}', encoding="utf-8")
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(args.threads)
    model_path = model_location("BAAI/bge-m3", ROOT / "tmp/models", False)
    print("Loading local BGE-M3...", flush=True)
    encoder = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    corpus_path = ROOT / "runtime/points_cache.jsonl"
    legacy = OfflineCorpus(corpus_path)
    # Verify legacy vectors with the exact original document-prefix pipeline.
    from backend.app.ingest.extract import _embedding_prefix
    indices = [0, len(legacy.rows) // 2, len(legacy.rows) - 1]
    checks = encode_documents(encoder, [f"{_embedding_prefix(legacy.rows[i]['payload'])} {legacy.rows[i]['payload']['text']}"
                                      for i in indices], show_progress_bar=False)
    cosines = [float(np.dot(vector, legacy.vectors[i])) for vector, i in zip(checks, indices)]
    if min(cosines) < .995:
        raise ValueError(f"Legacy vectors incompatible: {cosines}")
    urls = {r["payload"]["source"]: source_urls(ROOT, r["payload"]["source"]) for r in legacy.rows}
    cards = [{"id": r["id"], "payload": make_card(r["payload"], urls[r["payload"]["source"]])}
             for r in legacy.rows]
    # Checkpointed cache is keyed by exact card texts + model revision + code;
    # a rerun cannot silently reuse vectors after changing the representation.
    fingerprint = hashlib.sha256(json.dumps({"cards": cards, "model": model_path,
        "embedding_code": (ROOT / "backend/app/providers/embeddings/sentence_transformer.py").read_text(encoding="utf-8")},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    vector_path = args.output / f"card-vectors-{fingerprint}.jsonl"
    cached = {}
    if vector_path.exists():
        for line in vector_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cached[str(row["id"])] = row["vector"]
    tick = perf_counter()
    with vector_path.open("a", encoding="utf-8") as cache:
        for offset in range(0, len(cards), 4):
            batch = [r for r in cards[offset:offset + 4] if str(r["id"]) not in cached]
            if batch:
                vectors = encode_documents(encoder, [r["payload"]["text"] for r in batch],
                                           batch_size=4, show_progress_bar=False)
                for row, vector in zip(batch, vectors):
                    cached[str(row["id"])] = vector.tolist()
                    cache.write(json.dumps({"id": row["id"], "vector": vector.tolist()}) + "\n")
                cache.flush()
            print(f"Cards embedded: {min(offset + 4, len(cards))}/{len(cards)}", flush=True)
    indexing_ms = (perf_counter() - tick) * 1000
    for row in cards:
        row["vector"] = cached[str(row["id"])]
    card_path = args.output / "cards.jsonl"
    card_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cards), encoding="utf-8")
    structured = OfflineCorpus(card_path)
    dataset = ROOT / "eval/golden.jsonl"
    cases = read_cases(dataset)
    modes = {"hybrid": (legacy, False), "hybrid_fanout": (legacy, True),
             "cards": (structured, False), "cards_fanout": (structured, True)}
    for corpus, fanout in modes.values():
        measure(cases[0], corpus, encoder, fanout, args.candidates)
    rows, rng = [], random.Random(42)
    with (args.output / "runs.jsonl").open("w", encoding="utf-8") as out:
        for repeat in range(args.repeats):
            order = list(cases)
            rng.shuffle(order)
            for i, case in enumerate(order, 1):
                mode_order = list(modes)
                rng.shuffle(mode_order)
                for mode in mode_order:
                    corpus, fanout = modes[mode]
                    row = {**measure(case, corpus, encoder, fanout, args.candidates),
                           "mode": mode, "repeat": repeat}
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                print(f"Repeat {repeat + 1}: {i}/{len(cases)}", flush=True)
    groups = {mode: [r for r in rows if r["mode"] == mode] for mode in modes}
    report = {"complete": True, "status": "completed", "cases": len(cases),
        "settings": vars(args), "model": model_path, "legacy_vector_cosines": cosines,
        "card_fingerprint": fingerprint, "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "indexing_ms_this_run": indexing_ms, "package_versions": package_versions(),
        "cards": len(cards), "cards_with_source_urls": sum(bool(r["payload"]["source_urls"]) for r in cards),
        "typed_fields": sum("key" in f for r in cards for f in r["payload"]["fields"]),
        "summary": {mode: summarize(group) for mode, group in groups.items()},
        "by_category": {mode: {cat: summarize([r for r in group if r["category"] == cat])
            for cat in sorted({r["category"] for r in group})} for mode, group in groups.items()},
        "by_split": {mode: {split: summarize([r for r in group if r["split"] == split])
            for split in ("calibration", "holdout")} for mode, group in groups.items()},
        "notes": ["CPU exact cosine, no Qdrant/network/LLM latency; loading/indexing/warmup excluded.",
            "Same 319 chunk boundaries and evidence labels. No answer correctness measured.",
            "Fan-out is sequential, includes every query embedding, balances at most 5 results.",
            "Numeric fields are exported, not used for filtering/routing in this ablation.",
            "No applicability/exception resolution or general-policy expansion implemented.",
            "Known pilot dataset: no tuning in this run; not an independent unseen evaluation."]}
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
