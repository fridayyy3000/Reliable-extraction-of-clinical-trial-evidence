#!/usr/bin/env python3
"""Paper-clustered bootstrap CIs from the validated canonical cell dataset."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "analysis" / "journal" / "canonical_cell_level_results.csv"
OUTPUT_DIR = ROOT / "analysis" / "journal"
MODELS = ("Qwen", "Mistral", "Gemma", "Llama")
EXPECTED_POINT_ESTIMATES = {
    "Qwen": {"b1_accuracy_pct": 87.63, "evisearch_accuracy_pct": 91.26,
             "agreement_error_pct": 8.55, "disagreement_error_pct": 25.60,
             "review_error_recall_pct": 38.93},
    "Mistral": {"b1_accuracy_pct": 79.17, "evisearch_accuracy_pct": 81.52,
                "agreement_error_pct": 11.59, "disagreement_error_pct": 38.25,
                "review_error_recall_pct": 72.05},
    "Gemma": {"b1_accuracy_pct": 88.63, "evisearch_accuracy_pct": 83.70,
              "agreement_error_pct": 6.26, "disagreement_error_pct": 41.46,
              "review_error_recall_pct": 79.34},
    "Llama": {"b1_accuracy_pct": 76.65, "evisearch_accuracy_pct": 77.61,
              "agreement_error_pct": 9.94, "disagreement_error_pct": 40.09,
              "review_error_recall_pct": 81.12},
}
SEED = 20260927
REPLICATES = 10_000
PAPER_COUNT = 10
CELLS_PER_PAPER = 133


def parse_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be True or False, got {value!r}")


def load_rows() -> dict[str, dict[str, list[dict[str, Any]]]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5_320:
        raise ValueError(f"Expected 5,320 canonical rows, found {len(rows)}")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: defaultdict(list) for model in MODELS
    }
    seen: set[tuple[str, str, str]] = set()
    for raw in rows:
        model = raw["model"]
        if model not in grouped:
            raise ValueError(f"Unexpected model: {model}")
        key = (model, raw["paper_id"], raw["column_name"])
        if key in seen:
            raise ValueError(f"Duplicate model/paper/column row: {key}")
        seen.add(key)
        row = dict(raw)
        for field in ("b1_accuracy", "evisearch_accuracy"):
            row[field] = float(raw[field])
        row["a_b_agreement"] = parse_bool(raw["a_b_agreement"], "a_b_agreement")
        row["evisearch_binary_error"] = parse_bool(raw["evisearch_binary_error"], "evisearch_binary_error")
        row["review_flag"] = parse_bool(raw["review_flag"], "review_flag")
        grouped[model][raw["paper_id"]].append(row)

    for model, papers in grouped.items():
        if len(papers) != PAPER_COUNT:
            raise ValueError(f"{model}: expected 10 papers, found {len(papers)}")
        if any(len(cells) != CELLS_PER_PAPER for cells in papers.values()):
            counts = {paper: len(cells) for paper, cells in papers.items()}
            raise ValueError(f"{model}: expected 133 cells per paper, got {counts}")
    return grouped


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def point_estimates(papers: dict[str, list[dict[str, Any]]]) -> dict[str, float | int | None]:
    rows = [row for cells in papers.values() for row in cells]
    agreement = [row for row in rows if row["a_b_agreement"]]
    disagreement = [row for row in rows if not row["a_b_agreement"]]
    errors = [row for row in rows if row["evisearch_binary_error"]]
    flagged_errors = [row for row in errors if row["review_flag"]]
    flagged = [row for row in rows if row["review_flag"]]
    unflagged = [row for row in rows if not row["review_flag"]]
    agreement_error = safe_rate(sum(row["evisearch_binary_error"] for row in agreement), len(agreement))
    disagreement_error = safe_rate(sum(row["evisearch_binary_error"] for row in disagreement), len(disagreement))
    flagged_error = safe_rate(sum(row["evisearch_binary_error"] for row in flagged), len(flagged))
    unflagged_error = safe_rate(sum(row["evisearch_binary_error"] for row in unflagged), len(unflagged))
    return {
        "b1_accuracy_pct": 100 * sum(row["b1_accuracy"] for row in rows) / len(rows),
        "evisearch_accuracy_pct": 100 * sum(row["evisearch_accuracy"] for row in rows) / len(rows),
        "e_minus_b1_pp": 100 * (sum(row["evisearch_accuracy"] - row["b1_accuracy"] for row in rows) / len(rows)),
        "agreement_error_pct": None if agreement_error is None else 100 * agreement_error,
        "disagreement_error_pct": None if disagreement_error is None else 100 * disagreement_error,
        "disagreement_minus_agreement_pp": (
            None if agreement_error is None or disagreement_error is None
            else 100 * (disagreement_error - agreement_error)
        ),
        "disagreement_risk_ratio": (
            None if not agreement_error else disagreement_error / agreement_error
        ),
        "review_error_recall_pct": (
            None if not errors else 100 * len(flagged_errors) / len(errors)
        ),
        "flagged_error_pct": None if flagged_error is None else 100 * flagged_error,
        "unflagged_error_pct": None if unflagged_error is None else 100 * unflagged_error,
        "flagged_minus_unflagged_pp": (
            None if flagged_error is None or unflagged_error is None
            else 100 * (flagged_error - unflagged_error)
        ),
        "flagged_unflagged_risk_ratio": (
            None if not unflagged_error else flagged_error / unflagged_error
        ),
        "agreement_n": len(agreement),
        "disagreement_n": len(disagreement),
        "flagged_n": len(flagged),
        "unflagged_n": len(unflagged),
        "evisearch_error_n": len(errors),
    }


def validate_point_estimates(points: dict[str, dict[str, Any]]) -> None:
    fields = ("b1_accuracy_pct", "evisearch_accuracy_pct", "agreement_error_pct",
              "disagreement_error_pct", "review_error_recall_pct")
    problems = []
    for model in MODELS:
        for field in fields:
            actual = points[model][field]
            expected = EXPECTED_POINT_ESTIMATES[model][field]
            if actual is None or round(actual + 1e-12, 2) != expected:
                problems.append(f"{model} {field}: expected {expected:.2f}, got {actual}")
    if problems:
        raise ValueError("Canonical point-estimate validation failed; bootstrap stopped.\n" + "\n".join(problems))


def paper_statistics(cells: list[dict[str, Any]]) -> dict[str, int | float]:
    agreement = [row for row in cells if row["a_b_agreement"]]
    disagreement = [row for row in cells if not row["a_b_agreement"]]
    flagged = [row for row in cells if row["review_flag"]]
    unflagged = [row for row in cells if not row["review_flag"]]
    return {
        "n": len(cells),
        "b1_sum": sum(row["b1_accuracy"] for row in cells),
        "e_sum": sum(row["evisearch_accuracy"] for row in cells),
        "agree_n": len(agreement),
        "agree_errors": sum(row["evisearch_binary_error"] for row in agreement),
        "disagree_n": len(disagreement),
        "disagree_errors": sum(row["evisearch_binary_error"] for row in disagreement),
        "error_n": sum(row["evisearch_binary_error"] for row in cells),
        "flagged_n": len(flagged),
        "flagged_errors": sum(row["evisearch_binary_error"] for row in flagged),
        "unflagged_n": len(unflagged),
        "unflagged_errors": sum(row["evisearch_binary_error"] for row in unflagged),
    }


def replicate_metrics(sampled_papers: list[dict[str, int | float]]) -> dict[str, float | None]:
    totals: Counter[str] = Counter()
    for stats in sampled_papers:
        totals.update(stats)
    b1 = totals["b1_sum"] / totals["n"]
    e = totals["e_sum"] / totals["n"]
    agree_error = safe_rate(int(totals["agree_errors"]), int(totals["agree_n"]))
    disagree_error = safe_rate(int(totals["disagree_errors"]), int(totals["disagree_n"]))
    flagged_error = safe_rate(int(totals["flagged_errors"]), int(totals["flagged_n"]))
    unflagged_error = safe_rate(int(totals["unflagged_errors"]), int(totals["unflagged_n"]))
    recall = safe_rate(int(totals["flagged_errors"]), int(totals["error_n"]))
    return {
        "b1_accuracy_pct": 100 * b1,
        "evisearch_accuracy_pct": 100 * e,
        "e_minus_b1_pp": 100 * (e - b1),
        "agreement_error_pct": None if agree_error is None else 100 * agree_error,
        "disagreement_error_pct": None if disagree_error is None else 100 * disagree_error,
        "disagreement_minus_agreement_pp": (
            None if agree_error is None or disagree_error is None
            else 100 * (disagree_error - agree_error)
        ),
        "disagreement_risk_ratio": (
            None if agree_error in (None, 0) or disagree_error is None
            else disagree_error / agree_error
        ),
        "review_error_recall_pct": None if recall is None else 100 * recall,
        "flagged_error_pct": None if flagged_error is None else 100 * flagged_error,
        "unflagged_error_pct": None if unflagged_error is None else 100 * unflagged_error,
        "flagged_minus_unflagged_pp": (
            None if flagged_error is None or unflagged_error is None
            else 100 * (flagged_error - unflagged_error)
        ),
        "flagged_unflagged_risk_ratio": (
            None if unflagged_error in (None, 0) or flagged_error is None
            else flagged_error / unflagged_error
        ),
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


BOOTSTRAP_METRICS = (
    "b1_accuracy_pct", "evisearch_accuracy_pct", "e_minus_b1_pp",
    "agreement_error_pct", "disagreement_error_pct", "disagreement_minus_agreement_pp",
    "disagreement_risk_ratio", "review_error_recall_pct", "flagged_minus_unflagged_pp",
    "flagged_unflagged_risk_ratio",
)


def run_bootstrap(grouped: dict[str, dict[str, list[dict[str, Any]]]],
                  points: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rng = random.Random(SEED)
    result = {}
    for model in MODELS:
        papers = sorted(grouped[model])
        stats = {paper: paper_statistics(grouped[model][paper]) for paper in papers}
        draws: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
        undefined = Counter()
        for _ in range(REPLICATES):
            # One paper sample supplies every paired metric in this replicate.
            selected = rng.choices(papers, k=PAPER_COUNT)
            values = replicate_metrics([stats[paper] for paper in selected])
            for metric in BOOTSTRAP_METRICS:
                value = values[metric]
                if value is None:
                    undefined[metric] += 1
                else:
                    draws[metric].append(value)
        result[model] = {
            "point_estimates": points[model],
            "bootstrap": {
                metric: {
                    "ci_95": [percentile(draws[metric], 0.025), percentile(draws[metric], 0.975)],
                    "valid_replicates": len(draws[metric]),
                    "undefined_replicates": undefined[metric],
                }
                for metric in BOOTSTRAP_METRICS
            },
        }
    return result


def format_ci(point: float, interval: list[float | None], digits: int = 2) -> str:
    low, high = interval
    if low is None or high is None:
        return f"{point:.{digits}f} (NA)"
    return f"{point:.{digits}f} ({low:.{digits}f}, {high:.{digits}f})"


def build_table(results: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    csv_rows = []
    markdown_rows = []
    for model in MODELS:
        points = results[model]["point_estimates"]
        boot = results[model]["bootstrap"]
        row = {"model": model}
        row["b1_accuracy_pct"] = points["b1_accuracy_pct"]
        row["b1_accuracy_ci_low_pct"], row["b1_accuracy_ci_high_pct"] = boot["b1_accuracy_pct"]["ci_95"]
        row["evisearch_accuracy_pct"] = points["evisearch_accuracy_pct"]
        row["evisearch_accuracy_ci_low_pct"], row["evisearch_accuracy_ci_high_pct"] = boot["evisearch_accuracy_pct"]["ci_95"]
        for metric, ci in (("e_minus_b1_pp", "e_minus_b1_pp"),
                           ("agreement_error_pct", "agreement_error_pct"),
                           ("disagreement_error_pct", "disagreement_error_pct"),
                           ("disagreement_minus_agreement_pp", "disagreement_minus_agreement_pp"),
                           ("disagreement_risk_ratio", "disagreement_risk_ratio"),
                           ("review_error_recall_pct", "review_error_recall_pct"),
                           ("flagged_error_pct", None),
                           ("unflagged_error_pct", None),
                           ("flagged_minus_unflagged_pp", "flagged_minus_unflagged_pp"),
                           ("flagged_unflagged_risk_ratio", "flagged_unflagged_risk_ratio")):
            row[metric] = points[metric]
            if ci:
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = boot[ci]["ci_95"]
        row["disagreement_risk_ratio_undefined_replicates"] = boot["disagreement_risk_ratio"]["undefined_replicates"]
        row["flagged_unflagged_risk_ratio_undefined_replicates"] = boot["flagged_unflagged_risk_ratio"]["undefined_replicates"]
        csv_rows.append(row)

        def interval(metric: str, estimate_key: str, digits: int = 2) -> str:
            ci = boot[metric]["ci_95"]
            return format_ci(float(points[estimate_key]), ci, digits)

        cells = [
            model,
            interval("b1_accuracy_pct", "b1_accuracy_pct"),
            interval("evisearch_accuracy_pct", "evisearch_accuracy_pct"),
            interval("e_minus_b1_pp", "e_minus_b1_pp"),
            interval("agreement_error_pct", "agreement_error_pct"),
            interval("disagreement_error_pct", "disagreement_error_pct"),
            interval("disagreement_minus_agreement_pp", "disagreement_minus_agreement_pp"),
            interval("disagreement_risk_ratio", "disagreement_risk_ratio"),
            interval("review_error_recall_pct", "review_error_recall_pct"),
            f"{points['flagged_error_pct']:.2f}",
            f"{points['unflagged_error_pct']:.2f}",
            interval("flagged_unflagged_risk_ratio", "flagged_unflagged_risk_ratio"),
        ]
        markdown_rows.append("| " + " | ".join(cells) + " |")
    return csv_rows, markdown_rows


def write_outputs(results: dict[str, dict[str, Any]], validation_points: dict[str, dict[str, Any]]) -> None:
    csv_rows, markdown_rows = build_table(results)
    csv_path = OUTPUT_DIR / "clinical_bootstrap_ci.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = OUTPUT_DIR / "bootstrap_ci_replicates_summary.json"
    json_payload = {
        "design": {
            "resampling_unit": "paper/publication",
            "papers_per_replicate": PAPER_COUNT,
            "cells_per_paper": CELLS_PER_PAPER,
            "replicates": REPLICATES,
            "seed": SEED,
            "percentile_method": "linear interpolation at (n-1)*p",
            "point_estimate_validation": "PASS",
            "input": str(DATA_PATH.relative_to(ROOT)),
            "qwen_b1_failed_cell_note": "Four Extraction error cells retain frozen 1/1 scores because gold and scorer-normalized prediction were both empty.",
        },
        "point_estimate_validation": validation_points,
        "results": results,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    headers = [
        "Model", "B1 Accuracy % (95% CI)", "EviSearch Accuracy % (95% CI)",
        "E−B1 pp (95% CI)", "Agreement Error % (95% CI)",
        "Disagreement Error % (95% CI)", "Disagreement−Agreement pp (95% CI)",
        "Disagreement Risk Ratio (95% CI)", "Review Error Recall % (95% CI)",
        "Flagged Error %", "Unflagged Error %", "Flagged/Unflagged Risk Ratio (95% CI)",
    ]
    lines = [
        "# Clinical Bootstrap Confidence Intervals", "",
        f"Paper-clustered percentile bootstrap: {REPLICATES:,} replicates; fixed seed {SEED}; 10 papers sampled with replacement per replicate; all 133 cells retained per sampled paper.",
        "Point-estimate validation against the frozen manuscript values: **PASS**. No inference, extraction, verification, or judging was run.", "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *markdown_rows, "",
        "## Flagged vs Unflagged Error Enrichment", "",
        "| Model | Flagged−Unflagged Error Difference, pp (95% CI) | Disagreement/Agreement RR Undefined Replicates | Flagged/Unflagged RR Undefined Replicates |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        row = next(item for item in csv_rows if item["model"] == model)
        difference = format_ci(
            float(row["flagged_minus_unflagged_pp"]),
            [row["flagged_minus_unflagged_pp_ci_low"], row["flagged_minus_unflagged_pp_ci_high"]],
        )
        lines.append(
            f"| {model} | {difference} | {row['disagreement_risk_ratio_undefined_replicates']} | "
            f"{row['flagged_unflagged_risk_ratio_undefined_replicates']} |"
        )
    lines.extend([
        "",
        "The primary analysis uses canonical frozen scores unchanged, including the four Qwen B1 extraction-error cells scored 1/1 by the frozen empty/empty rule.",
        "",
    ])
    (OUTPUT_DIR / "clinical_bootstrap_ci.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    grouped = load_rows()
    points = {model: point_estimates(grouped[model]) for model in MODELS}
    validate_point_estimates(points)
    results = run_bootstrap(grouped, points)
    write_outputs(results, points)
    print("Point-estimate validation: PASS")
    print(f"Bootstrap complete: {REPLICATES} paper-clustered replicates/model, seed {SEED}")
    for model in MODELS:
        print(f"{model}: B1 {points[model]['b1_accuracy_pct']:.2f}, EviSearch {points[model]['evisearch_accuracy_pct']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())