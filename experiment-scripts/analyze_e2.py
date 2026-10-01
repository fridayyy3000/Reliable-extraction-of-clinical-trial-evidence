#!/usr/bin/env python3
"""Offline E2 agreement, reconciliation, and review-signal analysis for scored E1 outputs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import claude_scoring as cs  # noqa: E402


def system_map() -> Dict[str, cs.System]:
    return {
        "A": cs.System.parse("Mistral-A=e1-mistral-small32-full-a-r1/agent_extractor"),
        "B": cs.System.parse("Mistral-B=e1-mistral-small32-full-e-r1/search_agent"),
        "E": cs.System.parse("Mistral-E=e1-mistral-small32-full-e-r1/reconciliation_agent"),
    }


def scored_map(system: cs.System, docs: Sequence[str], labels: Dict[str, dict]) -> Dict[tuple[str, str], cs.Scored]:
    scored, unscored = cs.score(cs.cells(system, docs), labels)
    if unscored:
        raise ValueError(f"{system.name}: {len(unscored)} cells are unscored")
    return {(row.cell.doc, row.cell.column): row for row in scored}


def build_rows(docs: Sequence[str], labels: Dict[str, dict]) -> List[Dict[str, Any]]:
    systems = system_map()
    maps = {key: scored_map(system, docs, labels) for key, system in systems.items()}
    keys = list(maps["E"])
    if any(set(mapping) != set(keys) for mapping in maps.values()):
        raise ValueError("A, B, and E do not cover the same document/column cells")
    rows = []
    for key in keys:
        a, b, e = (maps[name][key] for name in ("A", "B", "E"))
        if e.cell.needs_review is None:
            raise ValueError(f"{key}: E needs_review is missing")
        agree = bool(e.cell.agents_agree)
        a_full, b_full, e_full = a.score == 1.0, b.score == 1.0, e.score == 1.0
        if agree:
            outcome = "agreement"
        elif a_full and not b_full:
            outcome = "A only correct"
        elif b_full and not a_full:
            outcome = "B only correct"
        elif a_full and b_full:
            outcome = "both correct"
        else:
            outcome = "both wrong"
        rows.append({
            "doc": key[0], "column": key[1], "category": e.cell.category,
            "a_pred": a.cell.pred, "b_pred": b.cell.pred, "e_pred": e.cell.pred,
            "agree": agree, "outcome": outcome, "needs_review": bool(e.cell.needs_review),
            "verification": e.cell.verification or "",
            "a_correctness": a.correctness, "a_completeness": a.completeness, "a_score": a.score,
            "b_correctness": b.correctness, "b_completeness": b.completeness, "b_score": b.score,
            "e_correctness": e.correctness, "e_completeness": e.completeness, "e_score": e.score,
            "a_fully_correct": a_full, "b_fully_correct": b_full, "e_fully_correct": e_full,
            "e_error": not e_full,
        })
    return rows


def pct(n: float, d: int) -> float | None:
    return round(100 * n / d, 2) if d else None


def summary(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(group)
    return {
        "count": n,
        "a_score_sum": round(sum(r["a_score"] for r in group), 3),
        "a_accuracy": pct(sum(r["a_score"] for r in group), n),
        "b_score_sum": round(sum(r["b_score"] for r in group), 3),
        "b_accuracy": pct(sum(r["b_score"] for r in group), n),
        "e_score_sum": round(sum(r["e_score"] for r in group), 3),
        "e_accuracy": pct(sum(r["e_score"] for r in group), n),
        "e_fully_correct_count": sum(r["e_fully_correct"] for r in group),
        "e_fully_correct_rate": pct(sum(r["e_fully_correct"] for r in group), n),
        "e_error_count": sum(r["e_error"] for r in group),
        "e_error_rate": pct(sum(r["e_error"] for r in group), n),
    }


def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float | None]:
    agree = [r for r in rows if r["agree"]]
    disagree = [r for r in rows if not r["agree"]]
    errors = [r for r in rows if r["e_error"]]
    return {
        "disagreement_rate": len(disagree) / len(rows) if rows else None,
        "e_accuracy_agreement": mean(r["e_score"] for r in agree) if agree else None,
        "e_accuracy_disagreement": mean(r["e_score"] for r in disagree) if disagree else None,
        "accuracy_difference_agreement_minus_disagreement": (
            mean(r["e_score"] for r in agree) - mean(r["e_score"] for r in disagree)
            if agree and disagree else None
        ),
        "review_flag_recall": (
            sum(r["needs_review"] for r in errors) / len(errors) if errors else None
        ),
        "e_error_rate_agreement": sum(r["e_error"] for r in agree) / len(agree) if agree else None,
        "e_error_rate_disagreement": sum(r["e_error"] for r in disagree) / len(disagree) if disagree else None,
    }


def clustered_bootstrap(rows: Sequence[Dict[str, Any]], docs: Sequence[str], replicates: int,
                        seed: int) -> Dict[str, Dict[str, Any]]:
    by_doc = {doc: [r for r in rows if r["doc"] == doc] for doc in docs}
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {}
    for _ in range(replicates):
        sampled = [rng.choice(docs) for _ in docs]
        sample_rows = [r for doc in sampled for r in by_doc[doc]]
        for name, value in metrics(sample_rows).items():
            if value is not None:
                draws.setdefault(name, []).append(float(value))

    observed = metrics(rows)
    result = {}
    for name, value in observed.items():
        values = sorted(draws.get(name, []))
        if not values or value is None:
            result[name] = {"estimate": None, "ci_low": None, "ci_high": None, "replicates": len(values)}
            continue
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        result[name] = {
            "estimate": round(100 * float(value), 2),
            "ci_low": round(100 * lo, 2),
            "ci_high": round(100 * hi, 2),
            "replicates": len(values),
        }
    return result


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, table_a: List[Dict[str, Any]], bootstrap: Dict[str, Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = ["A = B", "A ≠ B"]
    values = [table_a[0]["e_error_rate"], table_a[1]["e_error_rate"]]
    ci = [bootstrap["e_error_rate_agreement"], bootstrap["e_error_rate_disagreement"]]
    lower = [v - c["ci_low"] for v, c in zip(values, ci)]
    upper = [c["ci_high"] - v for v, c in zip(values, ci)]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    bars = ax.bar(labels, values, color=["#4C78A8", "#E45756"], yerr=[lower, upper], capsize=5)
    ax.set_ylabel("EviSearch cells not fully correct (%)")
    ax.set_title("EviSearch error rate by agent agreement")
    ax.set_ylim(0, max(values + [c["ci_high"] for c in ci]) * 1.2)
    for bar, value, row in zip(bars, values, table_a):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{value:.1f}%\n({row['e_error_count']}/{row['count']})", ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_report(table_a: List[Dict[str, Any]], table_b: List[Dict[str, Any]],
                    table_c: List[Dict[str, Any]], bootstrap: Dict[str, Dict[str, Any]],
                    total: int, disagreement: int, seed: int, replicates: int) -> str:
    def md_table(rows: List[Dict[str, Any]], columns: List[tuple[str, str]]) -> str:
        head = "| " + " | ".join(label for _, label in columns) + " |"
        sep = "|" + "|".join("---" for _ in columns) + "|"
        body = ["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows]
        return "\n".join([head, sep, *body])

    minimum = ["disagreement_rate", "e_accuracy_agreement", "e_accuracy_disagreement",
               "accuracy_difference_agreement_minus_disagreement", "review_flag_recall"]
    cis = [{"metric": name, **bootstrap[name]} for name in minimum]
    return f"""# E2: Agent disagreement and EviSearch reconciliation

Offline analysis of {total} scored cells from 10 papers. Agent disagreement is descriptive, not a claim of calibrated
uncertainty. Accuracy is the existing graded score `(correctness + completeness) / 2`; an error is a cell whose score
is below 1 (not fully correct).

- Disagreement: {disagreement}/{total} ({pct(disagreement, total)}%)
- Bootstrap: paper-clustered, {replicates:,} replicates, seed `{seed}`

## Table A — Agreement stratification

{md_table(table_a, [('group','Group'),('count','Count'),('percent','%'),('a_accuracy','Agent A accuracy'),('b_accuracy','Agent B accuracy'),('e_accuracy','EviSearch accuracy'),('e_error_count','E errors'),('e_error_rate','E error rate')])}

## Table B — Disagreement outcome

{md_table(table_b, [('outcome','Outcome'),('count','Count'),('percent_of_disagreements','% disagreements'),('e_fully_correct_count','E fully correct'),('e_fully_correct_rate','E fully correct rate')])}

## Table C — Review signal

{md_table(table_c, [('review_group','Review group'),('count','Count'),('percent','%'),('e_accuracy','E accuracy'),('e_error_count','Errors'),('e_error_rate','Error rate')])}

## Paper-clustered bootstrap 95% confidence intervals

{md_table(cis, [('metric','Metric'),('estimate','Estimate'),('ci_low','95% CI low'),('ci_high','95% CI high'),('replicates','Replicates')])}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "experiment-scripts/scoring/labels.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/e2/mistral-small32")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    gold = cs.load_gold()
    docs = cs.select_docs("all", gold)
    labels = cs.load_labels(args.labels)
    rows = build_rows(docs, labels)
    if len(rows) != 1330:
        raise ValueError(f"expected 1330 cells, found {len(rows)}")

    agree = [r for r in rows if r["agree"]]
    disagree = [r for r in rows if not r["agree"]]
    table_a = []
    for name, group in (("A = B", agree), ("A != B", disagree)):
        table_a.append({"group": name, "percent": pct(len(group), len(rows)), **summary(group)})

    table_b = []
    for outcome in ("A only correct", "B only correct", "both correct", "both wrong"):
        group = [r for r in disagree if r["outcome"] == outcome]
        s = summary(group)
        table_b.append({"outcome": outcome, "percent_of_disagreements": pct(len(group), len(disagree)), **s})

    table_c = []
    for name, flag in (("flagged", True), ("unflagged", False)):
        group = [r for r in rows if r["needs_review"] is flag]
        table_c.append({"review_group": name, "percent": pct(len(group), len(rows)), **summary(group)})

    bootstrap = clustered_bootstrap(rows, docs, args.replicates, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cells.csv", rows)
    write_csv(args.output_dir / "table_a_agreement.csv", table_a)
    write_csv(args.output_dir / "table_b_disagreement_outcome.csv", table_b)
    write_csv(args.output_dir / "table_c_review_signal.csv", table_c)
    (args.output_dir / "bootstrap.json").write_text(json.dumps({
        "method": "paper-clustered percentile bootstrap", "papers": docs,
        "replicates": args.replicates, "seed": args.seed, "metrics": bootstrap,
    }, indent=2) + "\n")
    make_plot(args.output_dir / "evisearch_error_by_agreement.png", table_a, bootstrap)
    report = markdown_report(table_a, table_b, table_c, bootstrap, len(rows), len(disagree), args.seed, args.replicates)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "definitions": {"accuracy": "mean((correctness + completeness) / 2)",
                        "fully_correct": "correctness == 1 and completeness == 1",
                        "error": "not fully_correct", "agreement": "claude_scoring agents_agree"},
        "counts": {"papers": len(docs), "cells": len(rows), "agreement": len(agree), "disagreement": len(disagree)},
        "table_a": table_a, "table_b": table_b, "table_c": table_c, "bootstrap": bootstrap,
        "verification_counts": Counter(r["verification"] for r in rows),
    }, indent=2) + "\n")
    print(report)
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
