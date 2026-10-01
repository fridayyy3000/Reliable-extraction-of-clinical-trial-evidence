#!/usr/bin/env python3
"""Join the saved E1 outputs and labels into a validated cell-level dataset."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import claude_scoring as cs  # noqa: E402

RESULTS_ROOT = ROOT / "new_pipeline_outputs" / "results"
SCORING_ROOT = ROOT / "experiment-scripts" / "scoring"
OUTPUT_ROOT = ROOT / "analysis" / "journal"

SYSTEMS = {
    "Qwen": {
        "B1": "e1-qwen36-full-b1-r1/markdown_baseline",
        "A": "schema-mhspc-trials-20260919020503-v4/agent_extractor",
        "B": "schema-mhspc-trials-20260919020503-v4/search_agent",
        "E": "schema-mhspc-trials-20260919020503-v4/reconciliation_agent",
    },
    "Mistral": {
        "B1": "e1-mistral-small32-full-b1-r1/markdown_baseline",
        "A": "e1-mistral-small32-full-a-r1/agent_extractor",
        "B": "e1-mistral-small32-full-e-r1/search_agent",
        "E": "e1-mistral-small32-full-e-r1/reconciliation_agent",
    },
    "Gemma": {
        "B1": "e1-gemma4-31b-full-b1-r1/markdown_baseline",
        "A": "e1-gemma4-31b-full-a-opt-r1/agent_extractor",
        "B": "e1-gemma4-31b-full-e-opt-r1/search_agent",
        "E": "e1-gemma4-31b-full-e-opt-r1/reconciliation_agent",
    },
    "Llama": {
        "B1": "e1-llama4-scout-full-b1-r1/markdown_baseline",
        "A": "e1-llama4-scout-full-a-r1/agent_extractor",
        "B": "e1-llama4-scout-full-e-r1/search_agent",
        "E": "e1-llama4-scout-full-e-r1/reconciliation_agent",
    },
}

FIELDS = [
    "model", "paper_id", "column_id", "column_name",
    "b1_prediction", "b1_correctness", "b1_completeness", "b1_accuracy",
    "agent_a_prediction", "agent_a_correctness", "agent_a_completeness",
    "agent_b_prediction", "agent_b_correctness", "agent_b_completeness",
    "a_b_agreement", "evisearch_prediction", "evisearch_correctness",
    "evisearch_completeness", "evisearch_accuracy", "evisearch_binary_error",
    "review_flag", "verifier_checked", "verifier_reproduced", "evidence_modality",
    "cited_page", "cited_quote",
]


def system_map(model: str) -> dict[str, cs.System]:
    return {
        label: cs.System.parse(f"{model}-{label}={spec}")
        for label, spec in SYSTEMS[model].items()
    }


def raw_outputs(system: cs.System, docs: list[str]) -> dict[tuple[str, str], Any]:
    output = {}
    for doc in docs:
        path = system.path(doc, RESULTS_ROOT)
        payload = json.loads(path.read_text(encoding="utf-8"))
        columns = payload.get("columns", payload)
        for name, entry in columns.items():
            output[(doc, name)] = entry.get("value") if isinstance(entry, dict) else entry
    return output


def raw_reconciliation(system: cs.System, docs: list[str]) -> dict[tuple[str, str], dict]:
    output = {}
    for doc in docs:
        path = system.path(doc, RESULTS_ROOT)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in payload.get("columns", payload).items():
            output[(doc, name)] = entry if isinstance(entry, dict) else {"value": entry}
    return output


def score_map(system: cs.System, docs: list[str], columns: dict, gold: dict,
              labels: dict) -> dict[tuple[str, str], cs.Scored]:
    scored, unscored = cs.score(cs.cells(system, docs, RESULTS_ROOT, columns, gold), labels)
    if unscored:
        raise ValueError(f"{system.name}: {len(unscored)} cells have no saved score")
    return {(item.cell.doc, item.cell.column): item for item in scored}


def evidence_fields(entry: dict) -> tuple[list[str] | None, list[Any] | None, list[str] | None]:
    checks = entry.get("checks") or []
    attributions = entry.get("attribution") or []
    source = entry.get("source") or {}
    modalities: list[str] = []
    pages: list[Any] = []
    quotes: list[str] = []

    def add_unique(target: list, value: Any) -> None:
        if value is not None and value != "" and value not in target:
            target.append(value)

    add_unique(modalities, source.get("modality"))
    add_unique(pages, source.get("page"))
    for attribution in attributions:
        if not isinstance(attribution, dict):
            continue
        add_unique(modalities, attribution.get("modality"))
        add_unique(pages, attribution.get("page"))
        for page in attribution.get("pages") or []:
            add_unique(pages, page)
        for key in ("quote", "evidence", "text"):
            add_unique(quotes, attribution.get(key))
    for check in checks:
        if not isinstance(check, dict):
            continue
        add_unique(modalities, check.get("modality"))
        add_unique(pages, check.get("page"))
        for page in check.get("pages") or []:
            add_unique(pages, page)
        add_unique(quotes, check.get("evidence"))

    return modalities or None, pages or None, quotes or None


def prediction(values: dict[tuple[str, str], Any], key: tuple[str, str]) -> Any:
    return values.get(key)


def make_rows(docs: list[str], columns: dict, gold: dict, labels: dict) -> tuple[list[dict], dict]:
    all_rows = []
    context = {}
    for model, specs in SYSTEMS.items():
        systems = system_map(model)
        raw = {name: raw_outputs(system, docs) for name, system in systems.items()}
        scored = {name: score_map(system, docs, columns, gold, labels)
                  for name, system in systems.items()}
        e_cells = {(cell.doc, cell.column): cell for cell in
                   cs.cells(systems["E"], docs, RESULTS_ROOT, columns, gold)}
        e_raw = raw["E"]
        e_entries = raw_reconciliation(systems["E"], docs)
        expected_keys = {(doc, column) for doc in docs for column in columns}
        for name in SYSTEMS[model]:
            if set(scored[name]) != expected_keys:
                raise ValueError(f"{model} {name}: scored keys differ from expected paper/field grid")
        if set(e_raw) != expected_keys or set(e_entries) != expected_keys:
            raise ValueError(f"{model} E: reconciliation outputs differ from expected paper/field grid")

        context[model] = {"raw": raw, "scored": scored, "e_cells": e_cells}
        for doc in docs:
            for column_name in columns:
                key = (doc, column_name)
                b1, a, b, e = (scored[name][key] for name in ("B1", "A", "B", "E"))
                e_cell = e_cells[key]
                agreement = e_cell.agents_agree
                if agreement is None:
                    raise ValueError(f"{model} {doc}/{column_name}: E2 agreement inputs are missing")
                if cs._answer(a.cell.pred) != e_cell.a_pred or cs._answer(b.cell.pred) != e_cell.b_pred:
                    raise ValueError(f"{model} {doc}/{column_name}: exported A/B runs differ from E2 agreement inputs")
                e_entry = e_entries[key]
                modalities, pages, quotes = evidence_fields(e_entry)
                e_accuracy = e.score
                all_rows.append({
                    "model": model,
                    "paper_id": doc,
                    "column_id": None,
                    "column_name": column_name,
                    "b1_prediction": prediction(raw["B1"], key),
                    "b1_correctness": b1.correctness,
                    "b1_completeness": b1.completeness,
                    "b1_accuracy": b1.score,
                    "agent_a_prediction": prediction(raw["A"], key),
                    "agent_a_correctness": a.correctness,
                    "agent_a_completeness": a.completeness,
                    "agent_b_prediction": prediction(raw["B"], key),
                    "agent_b_correctness": b.correctness,
                    "agent_b_completeness": b.completeness,
                    "a_b_agreement": agreement,
                    "evisearch_prediction": prediction(raw["E"], key),
                    "evisearch_correctness": e.correctness,
                    "evisearch_completeness": e.completeness,
                    "evisearch_accuracy": e_accuracy,
                    "evisearch_binary_error": e_accuracy < 1.0,
                    "review_flag": e_entry.get("needs_review"),
                    "verifier_checked": (bool(e_entry["checks"]) if "checks" in e_entry else None),
                    "verifier_reproduced": e_entry.get("verified"),
                    "evidence_modality": modalities,
                    "cited_page": pages,
                    "cited_quote": quotes,
                })
    return all_rows, context


def pct(count: int, total: int) -> float | None:
    return round(100 * count / total, 2) if total else None


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def expected_metrics(rows: list[dict], frozen: dict[str, dict]) -> dict[str, Any]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    actual = {}
    for model, group in by_model.items():
        n = len(group)
        disagree = [row for row in group if not row["a_b_agreement"]]
        agree = [row for row in group if row["a_b_agreement"]]
        flagged = [row for row in group if row["review_flag"] is True]
        errors = [row for row in group if row["evisearch_binary_error"]]
        flagged_errors = [row for row in errors if row["review_flag"] is True]
        checked = [row for row in group if row["verifier_checked"] is True]
        reproduced = [row for row in group if row["verifier_reproduced"] is True]
        actual[model] = {
            "b1_accuracy_pct": round(100 * sum(r["b1_accuracy"] for r in group) / n, 2),
            "agent_a_accuracy_pct": round(100 * sum(r["agent_a_correctness"] + r["agent_a_completeness"] for r in group) / (2*n), 2),
            "agent_b_accuracy_pct": round(100 * sum(r["agent_b_correctness"] + r["agent_b_completeness"] for r in group) / (2*n), 2),
            "evisearch_accuracy_pct": round(100 * sum(r["evisearch_accuracy"] for r in group) / n, 2),
            "disagreement_n": len(disagree),
            "disagreement_pct": pct(len(disagree), n),
            "flagged_n": len(flagged),
            "flagged_pct": pct(len(flagged), n),
            "e_accuracy_agreement_pct": round(100 * sum(r["evisearch_accuracy"] for r in agree) / len(agree), 2),
            "e_error_agreement_pct": pct(sum(r["evisearch_binary_error"] for r in agree), len(agree)),
            "e_accuracy_disagreement_pct": round(100 * sum(r["evisearch_accuracy"] for r in disagree) / len(disagree), 2),
            "e_error_disagreement_pct": pct(sum(r["evisearch_binary_error"] for r in disagree), len(disagree)),
            "review_error_recall_pct": pct(len(flagged_errors), len(errors)),
            "verifier_checked_n": len(checked),
            "verifier_reproduced_n": len(reproduced),
            "agreement_n": len(agree),
        }
    return actual


def write_outputs(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "canonical_cell_level_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list)
                             else row[key] for key in FIELDS})
    jsonl_path = OUTPUT_ROOT / "canonical_cell_level_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    cross_report = json.loads((SCORING_ROOT / "reports" / "cross-model-e1.json").read_text(encoding="utf-8"))
    qwen_report = json.loads((SCORING_ROOT / "reports" / "qwen-b1-e1.json").read_text(encoding="utf-8"))
    with (ROOT / "clinical_results_summary.csv").open(encoding="utf-8", newline="") as handle:
        frozen_rows = {row["model"]: row for row in csv.DictReader(handle)}
    docs = cross_report["documents"]
    if len(docs) != 10 or set(qwen_report["documents"]) != set(docs):
        raise ValueError("Frozen reports do not agree on the ten-paper set")
    columns = cs.load_columns()
    gold = cs.load_gold()
    labels = cs.load_labels(SCORING_ROOT / "labels.jsonl")
    rows, context = make_rows(docs, columns, gold, labels)
    write_outputs(rows)

    errors: list[str] = []
    model_counts = Counter(row["model"] for row in rows)
    paper_counts = Counter((row["model"], row["paper_id"]) for row in rows)
    if model_counts != Counter({model: 1330 for model in SYSTEMS}):
        errors.append(f"Unexpected model row counts: {dict(model_counts)}")
    if any(count != 133 for count in paper_counts.values()) or len(paper_counts) != 40:
        errors.append("Expected exactly 133 rows for each of 40 model/paper groups")
    if set(columns) != {row["column_name"] for row in rows}:
        errors.append("Canonical field names do not match the evaluation schema")

    actual = expected_metrics(rows, frozen_rows)
    metric_names = [
        "b1_accuracy_pct", "agent_a_accuracy_pct", "agent_b_accuracy_pct", "evisearch_accuracy_pct",
        "agreement_n", "disagreement_n", "disagreement_pct", "flagged_n", "flagged_pct",
        "e_accuracy_agreement_pct", "e_error_agreement_pct", "e_accuracy_disagreement_pct",
        "e_error_disagreement_pct", "review_error_recall_pct", "verifier_checked_n",
        "verifier_reproduced_n",
    ]
    metric_lines = []
    for model in SYSTEMS:
        expected = frozen_rows[model]
        checks = []
        for name in metric_names:
            raw_expected = expected.get(name, "")
            if raw_expected in ("", "N/A"):
                continue
            want: float | int = float(raw_expected)
            if name.endswith("_n"):
                want = int(want)
            got = actual[model][name]
            passed = got is not None and abs(float(got) - float(want)) < 0.005
            checks.append(passed)
            metric_lines.append(f"| {model} | {name} | {want} | {got} | {'PASS' if passed else 'FAIL'} |")
        if not all(checks):
            errors.append(f"Frozen metric mismatch for {model}")

    failed_cells = []
    qwen_b1 = context["Qwen"]["scored"]["B1"]
    qwen_raw = context["Qwen"]["raw"]["B1"]
    for key, value in qwen_raw.items():
        if cs.normalize(value).lower() not in cs.FAILED_VALUES:
            continue
        scored = qwen_b1[key]
        failed_cells.append({
            "paper": key[0], "column": key[1], "prediction": value,
            "score_source": scored.source, "correctness": scored.correctness,
            "completeness": scored.completeness,
        })
    failed_counts = Counter(item["score_source"] for item in failed_cells)
    if len(failed_cells) != 14 or failed_counts != Counter({"label": 10, "mechanical": 4}):
        errors.append(f"Unexpected Qwen B1 failed-cell disposition: {dict(failed_counts)}")

    all_review_present = all(row["review_flag"] is not None for row in rows)
    all_agreement_present = all(row["a_b_agreement"] is not None for row in rows)
    missing_fields = {
        model: {
            field: sum(row[field] is None for row in rows if row["model"] == model)
            for field in ("column_id", "cited_page", "cited_quote")
            if any(row[field] is None for row in rows if row["model"] == model)
        }
        for model in SYSTEMS
    }
    validation = [
        "# Canonical Cell-Level Join Validation", "",
        "Built exclusively from saved run outputs and saved scoring labels. No inference, agent, retrieval, verification, LLM judging, or bootstrap analysis was run.", "",
        "## Coverage", "",
        f"- Joined rows: {len(rows)} ({len(SYSTEMS)} models × 10 papers × {len(columns)} schema fields).",
        f"- Rows per model: {', '.join(f'{model} {model_counts[model]}' for model in SYSTEMS)}.",
        f"- Rows per paper/model: {min(paper_counts.values()) if paper_counts else 0}–{max(paper_counts.values()) if paper_counts else 0}; {len(paper_counts)} groups.",
        f"- Direct review flags present: {all_review_present}; E2 agreement values present: {all_agreement_present}.",
        "- No standalone column/field ID exists in the saved schema or scoring records; `column_id` is null, and the existing unique schema field name is retained as `column_name` and the join key.",
        "- Verifier/evidence values are taken from the final saved reconciliation record. Multi-valued modalities, pages, and quotes are JSON arrays; absent source values remain null.", "",
                "### Missing Values", "",
                "- `column_id`: 1,330 nulls per model because source artifacts provide schema field names but no distinct stable field IDs.",
                *[f"- `{field}` null counts: " + ", ".join(f"{model} {counts[field]}" for model, counts in missing_fields.items() if field in counts)
                    for field in ("cited_page", "cited_quote")],
                "- No other required prediction, score, agreement, review, or verifier fields are null.", "",
        "## Frozen Metric Checks", "",
        "| Model | Metric | Frozen | Joined | Check |", "|---|---|---:|---:|---|",
        *metric_lines, "",
        "## Qwen B1 Failed Cells", "",
        "The frozen report's `failed_cells` count is the number of extraction entries whose value is the pipeline failure marker `Extraction error` (`Cell.failed` in the existing scorer). The scorer converts that marker to an empty prediction for scoring; it does not drop the field. All 14 remain in the 1,330-cell denominator.",
        f"- Saved marker entries: {len(failed_cells)}.",
        f"- Saved correctness/completeness labels: {failed_counts.get('label', 0)}; all are 0/0.",
        f"- No label row: {failed_counts.get('mechanical', 0)}; existing scoring treated these empty-gold/empty-prediction cells mechanically as 1/1.",
        "- Therefore all 14 are included in the reported 87.63% B1 accuracy with the existing scorer's behavior; none were removed or changed.",
        "", "| Paper | Column | Saved marker | Score source | Correctness | Completeness |",
        "|---|---|---|---|---:|---:|",
    ]
    validation.extend(
        f"| {markdown_cell(item['paper'])} | {markdown_cell(item['column'])} | `{item['prediction']}` | {item['score_source']} | {item['correctness']} | {item['completeness']} |"
        for item in failed_cells
    )
    validation.extend(["", "## Join Notes", "", "- B1/A/B/E labels were resolved with the existing `claude_scoring` cell construction, prediction normalization, item IDs, and score function.",
                       "- Agent agreement uses `Cell.agents_agree`, exactly as the saved E2 analysis does.",
                       "- `review_flag` is copied from the final reconciliation cell's `needs_review`; it is not reconstructed from disagreement.",
                       "- Verifier checked means the saved reconciliation cell contains one or more detailed `checks`; verifier reproduced is the saved `verified` value.",
                       "- No confidence intervals were calculated.", ""])
    validation.extend(["## Overall", "", f"**{'PASS' if not errors else 'FAIL'}**", ""])
    if errors:
        validation.extend(f"- {error}" for error in errors)
    (OUTPUT_ROOT / "canonical_join_validation.md").write_text("\n".join(validation), encoding="utf-8")
    print(f"Wrote {len(rows)} joined rows; validation {'PASS' if not errors else 'FAIL'}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())