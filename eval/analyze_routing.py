"""Evaluate a BM25 early-exit gate offline; never change application settings.

Fit only on calibration cases, report holdout separately. This is retrieval
confidence, not an answer probability. Latencies are conservative simulations:
fallback includes a repeated BM25 pass and excludes answer generation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import numpy as np


def accepted(features, gate):
    return (features["candidate_count"] > 0
            and features["bm25_top_score"] >= gate["score_min"]
            and features["bm25_gap"] >= gate["gap_min"]
            and features["term_coverage"] >= gate["coverage_min"])


def supported(row):
    if not row["answerable"]:
        return False
    score = row["metrics"].get("section_evidence_at_5")
    return bool(score if score is not None else row["metrics"]["all_evidence_at_5"])


def fit_gate(rows, min_accepts=10):
    """No holdout row is allowed to influence gate selection."""
    if any(row["split"] != "calibration" for row in rows):
        raise ValueError("Gate fitting accepts calibration cases only")
    if not rows:
        return None
    scores = [row["features"]["bm25_top_score"] for row in rows]
    best, best_count = None, 0
    for score in sorted(set(float(np.percentile(scores, p)) for p in (0, 25, 50, 75, 90)), reverse=True):
        for gap in (0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0):
            for coverage in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
                gate = {"score_min": score, "gap_min": gap, "coverage_min": coverage}
                selected = [row for row in rows if accepted(row["features"], gate)]
                if len(selected) >= min_accepts and len(selected) > best_count and all(supported(row) for row in selected):
                    best, best_count = gate, len(selected)
    return best


def aggregate_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["mode"])].append(row)
    result = defaultdict(dict)
    for (case_id, mode), group in groups.items():
        if len({json.dumps(row["metrics"], sort_keys=True) for row in group}) != 1:
            raise ValueError(f"Non-deterministic retrieval metrics: {case_id}/{mode}")
        result[case_id][mode] = {**group[0], "timings": {
            "total_ms": mean(row["timings"]["total_ms"] for row in group)}}
    return result


def evaluate_gate(cases, gate, fallback, split):
    rows, costs, baseline_costs, early_errors = [], [], [], []
    early_count = negative_accepts = regressions = 0
    for case_id, modes in cases.items():
        bm, full = modes["bm25"], modes[fallback]
        if bm["split"] != split:
            continue
        early = gate is not None and accepted(bm["features"], gate)
        chosen = bm if early else full
        early_count += early
        if early and not supported(bm):
            early_errors.append(case_id)
        negative_accepts += bool(early and not bm["answerable"])
        if bm["answerable"]:
            regressions += bool(supported(full) and not supported(chosen))
            rows.append(chosen)
        costs.append(bm["timings"]["total_ms"] + (0 if early else full["timings"]["total_ms"]))
        baseline_costs.append(full["timings"]["total_ms"])
    annotated = [row for row in rows if row["metrics"].get("section_evidence_at_5") is not None]
    return {"cases": len(costs), "early_exits": early_count,
            "early_exit_rate": early_count / len(costs) if costs else None,
            "unsafe_early_exit_ids": early_errors, "unanswerable_early_exits": negative_accepts,
            "regressions_vs_fallback": regressions,
            "section_evidence_at_5": mean(row["metrics"]["section_evidence_at_5"] for row in annotated) if annotated else None,
            "all_sources_at_5": mean(row["metrics"]["all_evidence_at_5"] for row in rows) if rows else None,
            "simulated_p50_ms": float(np.percentile(costs, 50)) if costs else None,
            "simulated_p95_ms": float(np.percentile(costs, 95)) if costs else None,
            "fallback_p50_ms": float(np.percentile(baseline_costs, 50)) if costs else None,
            "fallback_p95_ms": float(np.percentile(baseline_costs, 95)) if costs else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--fallback", choices=["dense", "hybrid", "hybrid_rerank"], default="hybrid")
    parser.add_argument("--min-calibration-accepts", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = aggregate_cases(runs)
    if not cases or any("bm25" not in modes or args.fallback not in modes for modes in cases.values()):
        parser.error("Every case needs both BM25 and the selected fallback from the same comparison")
    calibration = [modes["bm25"] for modes in cases.values() if modes["bm25"]["split"] == "calibration"]
    gate = fit_gate(calibration, args.min_calibration_accepts)
    result = {"gate": gate, "fallback": args.fallback, "application_changed": False,
              "calibration": evaluate_gate(cases, gate, args.fallback, "calibration"),
              "holdout": evaluate_gate(cases, gate, args.fallback, "holdout"),
              "notes": ["Fitted for zero observed unsupported early exits on calibration only.",
                        "Small, partly template-generated set; not a production confidence guarantee.",
                        "Source-only labels used where section-level labels are absent.",
                        "Simulated sequential retrieval latency, not measured streaming TTFT.",
                        "Do not enable a gate with unsafe holdout exits or insufficient coverage."]}
    output = args.output or args.runs.with_name(f"routing-{args.fallback}.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
