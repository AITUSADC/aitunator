"""Export inspectable cards, example evidence context and paired metric changes."""
import argparse
import json
from pathlib import Path

from eval.run import ROOT
from backend.app.rag.structured import evidence_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "runtime/structured-experiment")
    args = parser.parse_args()
    directory = args.directory
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("Experiment has not completed")
    runs = [json.loads(line) for line in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    cards = [json.loads(line) for line in (directory / "cards.jsonl").read_text(encoding="utf-8").splitlines()]
    (directory / "evidence-cards.jsonl").write_text("".join(json.dumps({"id": r["id"], **r["payload"]},
        ensure_ascii=False) + "\n" for r in cards), encoding="utf-8")
    by_key = {(r["case_id"], r["mode"], r["repeat"]): r for r in runs}
    changes = {}
    for mode in ("hybrid_fanout", "cards", "cards_fanout"):
        gains, losses = [], []
        for row in runs:
            if row["mode"] != mode:
                continue
            old = by_key[row["case_id"], "hybrid", row["repeat"]]
            before, after = old["metrics"]["section_evidence_at_5"], row["metrics"]["section_evidence_at_5"]
            if before is None or before == after:
                continue
            (gains if after > before else losses).append({"case_id": row["case_id"], "repeat": row["repeat"],
                "question": row["question"], "split": row["split"], "category": row["category"]})
        changes[mode] = {"gains": gains, "losses": losses}
    (directory / "paired-changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    sample = next(r for r in runs if r["mode"] == "cards_fanout" and r["category"] == "comparison")
    lookup = {str(r["id"]): r["payload"] for r in cards}
    rows = [{**lookup[r["id"]], "_retrieval": {"id": r["id"]}} for r in sample["results"]]
    (directory / "example-context.md").write_text("# Пример найденных сведений\n\n"
        + sample["question"] + "\n\nЭто контекст поиска, не сгенерированный ответ.\n\n"
        + evidence_context(rows), encoding="utf-8")
    print(json.dumps({mode: {key: len(value) for key, value in result.items()}
                      for mode, result in changes.items()}, indent=2))


if __name__ == "__main__":
    main()
