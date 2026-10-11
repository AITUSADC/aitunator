"""Compare topic-aware selection with cached vectors; never rebuild document embeddings."""
import json
import os
import random
from collections import defaultdict
import hashlib
from pathlib import Path

from eval.run import ROOT, OfflineCorpus, read_cases, summarize, model_location, package_versions
from eval.structured_compare import measure


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DISABLE_SAFETENSORS_CONVERSION="true")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    output = ROOT / "runtime/section-experiment"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text('{"complete":false,"status":"running"}', encoding="utf-8")
    previous = json.loads((ROOT / "runtime/structured-experiment/summary.json").read_text(encoding="utf-8"))
    corpus_path = ROOT / "runtime/points_cache.jsonl"
    dataset = ROOT / "eval/golden.jsonl"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if not previous.get("complete") or previous["corpus_sha256"] != digest(corpus_path):
        raise ValueError("Need a completed compatible structured experiment")
    model = model_location("BAAI/bge-m3", ROOT / "tmp/models", False)
    if Path(model).name != Path(previous["model"]).name:
        raise ValueError("Model revision differs from card index")
    encoder = SentenceTransformer(model, device="cpu", local_files_only=True)
    legacy = OfflineCorpus(corpus_path)
    card_path = ROOT / "runtime/structured-experiment/cards.jsonl"
    cards = OfflineCorpus(card_path)
    modes = {"hybrid": (legacy, False, False), "hybrid_fanout": (legacy, True, False),
             "hybrid_sections": (legacy, True, True), "cards_fanout": (cards, True, False),
             "cards_sections": (cards, True, True)}
    cases = read_cases(dataset)
    for corpus, fanout, sections in modes.values():
        measure(cases[0], corpus, encoder, fanout, 8, sections)
    rows, rng = [], random.Random(42)
    rng.shuffle(cases)
    with (output / "runs.jsonl").open("w", encoding="utf-8") as out:
        for i, case in enumerate(cases, 1):
            order = list(modes)
            rng.shuffle(order)
            for mode in order:
                corpus, fanout, sections = modes[mode]
                row = {**measure(case, corpus, encoder, fanout, 8, sections), "mode": mode}
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            print(f"{i}/{len(cases)}", flush=True)
    groups = {m: [r for r in rows if r["mode"] == m] for m in modes}
    changes = {}
    for mode in ("hybrid_sections", "cards_sections"):
        baseline = {r["case_id"]: r for r in groups["hybrid"]}
        change = defaultdict(list)
        for row in groups[mode]:
            old = baseline[row["case_id"]]["metrics"]["section_evidence_at_5"]
            new = row["metrics"]["section_evidence_at_5"]
            if old is not None and old != new:
                change["gains" if new > old else "losses"].append(row["case_id"])
        changes[mode] = dict(change)
    report = {"complete": True, "status": "completed", "cases": len(cases),
        "settings": {"threads": 4, "device": "cpu", "candidates": 8, "top_k": 5, "repeats": 1, "seed": 42},
        "model": model, "package_versions": package_versions(),
        "corpus_sha256": digest(corpus_path), "cards_sha256": digest(card_path), "dataset_sha256": digest(dataset),
        "summary": {m: summarize(rs) for m, rs in groups.items()},
        "by_category": {m: {c: summarize([r for r in rs if r["category"] == c])
             for c in sorted({r["category"] for r in rs})} for m, rs in groups.items()},
        "by_split": {m: {s: summarize([r for r in rs if r["split"] == s])
             for s in ("calibration", "holdout")} for m, rs in groups.items()},
        "paired_changes_vs_hybrid": changes,
        "notes": ["Known pilot dataset, including templated comparisons; not an unseen evaluation.",
                  "Only query text and payload section_type drive selection; no evidence labels.",
                  "Stable partition of the same 8 candidates, same top-5 total, no hard filter.",
                  "Sequential fan-out; all query embeddings timed; no reranker, LLM or Qdrant."]}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
