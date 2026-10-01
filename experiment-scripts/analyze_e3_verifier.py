#!/usr/bin/env python3
"""Offline E3 verifier/reproduction reliability analysis for scored Mistral E1 outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation import claude_scoring as cs  # noqa: E402

RUN = "e1-mistral-small32-full-e-r1"
RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs/results"


def systems() -> dict[str, cs.System]:
    return {
        "A": cs.System.parse("Mistral-A=e1-mistral-small32-full-a-r1/agent_extractor"),
        "B": cs.System.parse("Mistral-B=e1-mistral-small32-full-e-r1/search_agent"),
        "E": cs.System.parse("Mistral-E=e1-mistral-small32-full-e-r1/reconciliation_agent"),
    }


def scored_map(system: cs.System, docs: Sequence[str], labels: dict[str, dict]) -> dict[tuple[str, str], cs.Scored]:
    scored, unscored = cs.score(cs.cells(system, docs), labels)
    if unscored:
        raise ValueError(f"{system.name}: {len(unscored)} cells are unscored")
    return {(x.cell.doc, x.cell.column): x for x in scored}


def load_verifier(docs: Sequence[str]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_cells: dict[tuple[str, str], dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    log_files: list[str] = []
    verifier_calls = 0
    missing_columns: list[tuple[str, str, str]] = []
    for doc in docs:
        agent_dir = RESULTS_ROOT / doc / "runs" / RUN / "reconciliation_agent"
        result_path = agent_dir / "reconciled_results.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        columns = payload["columns"]
        for column, value in columns.items():
            raw_cells[(doc, column)] = value
        for log_path in sorted((agent_dir / "verification_logs").glob("batch_*_conversation.json")):
            log_files.append(str(log_path.relative_to(PROJECT_ROOT)))
            log = json.loads(log_path.read_text(encoding="utf-8"))
            verifier_calls += len(log.get("verifier_calls") or [])
            for index, check in enumerate(log.get("checks") or []):
                column = check.get("column")
                if column not in columns:
                    missing_columns.append((doc, column, str(log_path)))
                    continue
                checks.append({
                    "doc": doc, "column": column,
                    "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                    "check_index": index, **check,
                })
    if missing_columns:
        raise ValueError(f"{len(missing_columns)} verifier checks do not map to result columns: {missing_columns[:3]}")
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for check in checks:
        by_key[(check["doc"], check["column"])].append(check)
    # The logs are the source of detailed checks; verify exact count/vocabulary against embedded copies.
    embedded = sum(len(x.get("checks") or []) for x in raw_cells.values())
    if embedded != len(checks):
        raise ValueError(f"log checks ({len(checks)}) != embedded checks ({embedded})")
    return raw_cells, checks, {
        "result_path_template": f"new_pipeline_outputs/results/<paper>/runs/{RUN}/reconciliation_agent/reconciled_results.json",
        "log_path_template": f"new_pipeline_outputs/results/<paper>/runs/{RUN}/reconciliation_agent/verification_logs/batch_*_conversation.json",
        "log_files": len(log_files), "verifier_calls": verifier_calls,
        "detailed_checks": len(checks), "unique_checked_cells": len(by_key),
        "verdict_vocabulary": dict(Counter(x.get("verdict") for x in checks)),
        "modality_vocabulary": dict(Counter(x.get("modality") for x in checks)),
        "all_checks_map_unambiguously": True,
    }


def verifier_state(raw: dict[str, Any], checks: Sequence[dict[str, Any]]) -> str:
    if not checks:
        return "not_checked"
    if raw.get("verified") is True:
        return "verified"
    verdicts = {x.get("verdict") for x in checks}
    if "partial" in verdicts:
        return "checked_partial"
    return "checked_not_reproduced"


def build_rows(docs: Sequence[str], labels: dict[str, dict]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    maps = {name: scored_map(system, docs, labels) for name, system in systems().items()}
    keys = set(maps["E"])
    if any(set(x) != keys for x in maps.values()):
        raise ValueError("A, B, and E cell sets differ")
    raw, checks, audit = load_verifier(docs)
    if set(raw) != keys:
        raise ValueError(f"raw reconciliation keys differ from scored E keys: {len(raw)} vs {len(keys)}")
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for check in checks:
        by_key[(check["doc"], check["column"])].append(check)
    rows = []
    for key in sorted(keys):
        a, b, e = (maps[x][key] for x in ("A", "B", "E"))
        r = raw[key]; cell_checks = by_key[key]
        if r.get("needs_review") is None or r.get("verified") is None or not r.get("verification"):
            raise ValueError(f"{key}: missing reconciliation metadata")
        state = verifier_state(r, cell_checks)
        modalities = sorted({x.get("modality") for x in cell_checks})
        rows.append({
            "doc": key[0], "column": key[1], "category": e.cell.category,
            "a_pred": a.cell.pred, "b_pred": b.cell.pred, "e_pred": e.cell.pred,
            "agents_agree": bool(e.cell.agents_agree), "needs_review": bool(r["needs_review"]),
            "reconciliation": r["verification"], "verified": bool(r["verified"]),
            "verifier_state": state, "verifier_checked": bool(cell_checks),
            "check_count": len(cell_checks), "modalities": ";".join(modalities),
            "supported_checks": sum(x.get("verdict") == "supported" for x in cell_checks),
            "partial_checks": sum(x.get("verdict") == "partial" for x in cell_checks),
            "not_supported_checks": sum(x.get("verdict") == "not_supported" for x in cell_checks),
            "e_correctness": e.correctness, "e_completeness": e.completeness,
            "e_graded_score": e.score, "e_fully_correct": e.score == 1.0, "e_error": e.score != 1.0,
        })
    cell_lookup = {(x["doc"], x["column"]): x for x in rows}
    for check in checks:
        row = cell_lookup[(check["doc"], check["column"])]
        check.update({
            "cell_verifier_state": row["verifier_state"], "cell_verified": row["verified"],
            "cell_graded_score": row["e_graded_score"], "cell_fully_correct": row["e_fully_correct"],
            "cell_error": row["e_error"],
        })
    audit["multi_check_cells"] = sum(x["check_count"] > 1 for x in rows)
    audit["checks_per_cell_distribution"] = dict(sorted(Counter(x["check_count"] for x in rows if x["check_count"]).items()))
    audit["verifier_state_counts"] = dict(Counter(x["verifier_state"] for x in rows))
    audit["reconciliation_vocabulary"] = dict(Counter(x["reconciliation"] for x in rows))
    return rows, checks, audit


def pct(n: float, d: int) -> float | None:
    return round(100 * n / d, 2) if d else None


def summarize(group: Sequence[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    n = len(group); errors = sum(x["e_error"] for x in group); full = n - errors
    return {
        "count": n, "percent": pct(n, total if total is not None else n),
        "graded_accuracy": pct(sum(x["e_graded_score"] for x in group), n),
        "fully_correct_count": full, "fully_correct_rate": pct(full, n),
        "error_count": errors, "error_rate": pct(errors, n),
    }


def grouped(rows: Sequence[dict[str, Any]], field: str, order: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    values = list(order) if order else sorted({x[field] for x in rows}, key=str)
    return [{field: value, **summarize([x for x in rows if x[field] == value], len(rows))} for value in values]


def comparison_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    checked = [x for x in rows if x["verifier_checked"]]
    rep = [x for x in checked if x["verified"]]
    non = [x for x in checked if not x["verified"]]
    er = sum(x["e_error"] for x in rep) / len(rep) if rep else None
    en = sum(x["e_error"] for x in non) / len(non) if non else None
    ar = mean(x["e_graded_score"] for x in rep) if rep else None
    an = mean(x["e_graded_score"] for x in non) if non else None
    return {
        "reproduction_rate_among_checked": len(rep) / len(checked) if checked else None,
        "error_rate_reproduced": er, "error_rate_not_reproduced": en,
        "error_rate_difference_not_reproduced_minus_reproduced": en - er if en is not None and er is not None else None,
        "graded_accuracy_difference_reproduced_minus_not_reproduced": ar - an if ar is not None and an is not None else None,
        "error_risk_ratio_not_reproduced_over_reproduced": en / er if en is not None and er not in (None, 0) else None,
    }


def bootstrap(rows: Sequence[dict[str, Any]], docs: Sequence[str], replicates: int, seed: int) -> dict[str, Any]:
    by_doc = {d: [x for x in rows if x["doc"] == d] for d in docs}
    observed = comparison_metrics(rows); draws: dict[str, list[float]] = defaultdict(list); rng = random.Random(seed)
    for _ in range(replicates):
        sample = [x for _ in docs for x in by_doc[rng.choice(docs)]]
        for name, value in comparison_metrics(sample).items():
            if value is not None and math.isfinite(value): draws[name].append(value)
    result = {}
    for name, estimate in observed.items():
        vals = sorted(draws[name])
        lo = vals[int(.025 * (len(vals) - 1))] if vals else None
        hi = vals[int(.975 * (len(vals) - 1))] if vals else None
        scale = 1 if "risk_ratio" in name else 100
        result[name] = {
            "estimate": round(scale * estimate, 3) if estimate is not None else None,
            "ci_low": round(scale * lo, 3) if lo is not None else None,
            "ci_high": round(scale * hi, 3) if hi is not None else None,
            "valid_replicates": len(vals), "requested_replicates": replicates,
            "unit": "ratio" if scale == 1 else "percent_or_percentage_points",
        }
    return result


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)


def md_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row.get(k, "")) for k, _ in columns) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def plot(path: Path, comp: dict[str, Any], boot: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    names = ["Reproduced", "Not reproduced"]
    values = [100 * comp["error_rate_reproduced"], 100 * comp["error_rate_not_reproduced"]]
    cis = [boot["error_rate_reproduced"], boot["error_rate_not_reproduced"]]
    yerr = [[v-c["ci_low"] for v,c in zip(values,cis)], [c["ci_high"]-v for v,c in zip(values,cis)]]
    fig, ax = plt.subplots(figsize=(6.4,4.4)); bars=ax.bar(names,values,color=["#4C78A8","#E45756"],yerr=yerr,capsize=5)
    ax.set_ylabel("EviSearch cells not fully correct (%)"); ax.set_title("EviSearch error rate by verifier outcome")
    ax.set_ylim(0,max([*values,*[c["ci_high"] for c in cis]])*1.22)
    for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.6,f"{v:.1f}%",ha="center")
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=PROJECT_ROOT/"experiment-scripts/scoring/labels.jsonl")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT/"experiments/e3/mistral-small32")
    p.add_argument("--replicates", type=int, default=10000); p.add_argument("--seed", type=int, default=20260927)
    args = p.parse_args(); docs=cs.select_docs("all",cs.load_gold()); labels=cs.load_labels(args.labels)
    rows, checks, audit = build_rows(docs, labels)
    if len(rows) != 1330: raise ValueError(f"expected 1330 cells, found {len(rows)}")
    states=["verified","checked_not_reproduced","checked_partial","not_checked"]
    table_a=grouped(rows,"verifier_state",states)
    checked=[x for x in rows if x["verifier_checked"]]; rep=[x for x in checked if x["verified"]]; non=[x for x in checked if not x["verified"]]
    comp=comparison_metrics(rows)
    table_b=[]
    for status, subset in (("reproduced",rep),("not_reproduced",non)):
        for flag in (False,True): table_b.append({"verifier_group":status,"needs_review":flag,**summarize([x for x in subset if x["needs_review"]==flag],len(rows))})
    table_c=[]
    for agree in (True,False):
        for status,subset in (("reproduced",rep),("not_reproduced",non),("not_checked",[x for x in rows if not x["verifier_checked"]])):
            table_c.append({"agents_agree":agree,"verifier_group":status,**summarize([x for x in subset if x["agents_agree"]==agree],len(rows))})
    table_d=[]
    for modality in sorted({x["modality"] for x in checks}):
        mc=[x for x in checks if x["modality"]==modality]; keys={(x["doc"],x["column"]) for x in mc}; cells=[x for x in rows if (x["doc"],x["column"]) in keys]
        table_d.append({"modality":modality,"detailed_checks":len(mc),"unique_cells":len(cells),
                        "supported_checks":sum(x["verdict"]=="supported" for x in mc),
                        "supported_check_rate":pct(sum(x["verdict"]=="supported" for x in mc),len(mc)),**summarize(cells)})
    reconciliation=[]
    for value in sorted({x["reconciliation"] for x in rows}):
        g=[x for x in rows if x["reconciliation"]==value]; gc=[x for x in g if x["verifier_checked"]]
        reconciliation.append({"reconciliation":value,"checked_count":len(gc),"checked_rate":pct(len(gc),len(g)),
                               "reproduced_count":sum(x["verified"] for x in gc),"reproduction_rate_among_checked":pct(sum(x["verified"] for x in gc),len(gc)),**summarize(g,len(rows))})
    categories=[]
    for value in ("exact_match","numeric_tolerance","structured_text"):
        g=[x for x in rows if x["category"]==value]; gc=[x for x in g if x["verifier_checked"]]; gr=[x for x in gc if x["verified"]]; gn=[x for x in gc if not x["verified"]]
        categories.append({"category":value,"count":len(g),"checked_count":len(gc),"checked_rate":pct(len(gc),len(g)),
                           "reproduced_count":len(gr),"reproduction_rate_among_checked":pct(len(gr),len(gc)),
                           "reproduced_error_count":sum(x["e_error"] for x in gr),"reproduced_error_rate":pct(sum(x["e_error"] for x in gr),len(gr)),
                           "not_reproduced_count":len(gn),"not_reproduced_error_count":sum(x["e_error"] for x in gn),"not_reproduced_error_rate":pct(sum(x["e_error"] for x in gn),len(gn))})
    mult=[]
    for n in sorted({x["check_count"] for x in rows}):
        g=[x for x in rows if x["check_count"]==n]; mult.append({"checks_per_cell":n,**summarize(g,len(rows))})
    per_paper=[]
    for doc in docs:
        g=[x for x in rows if x["doc"]==doc]; gc=[x for x in g if x["verifier_checked"]]
        per_paper.append({"paper":doc,"cells":len(g),"checked_cells":len(gc),"checked_rate":pct(len(gc),len(g)),"detailed_checks":sum(x["check_count"] for x in g)})
    boot=bootstrap(rows,docs,args.replicates,args.seed); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"cells.csv",rows); write_csv(out/"verifier_checks.csv",checks)
    write_csv(out/"table_a_verifier_state.csv",table_a); write_csv(out/"table_b_verifier_review.csv",table_b)
    write_csv(out/"table_c_verifier_disagreement.csv",table_c); write_csv(out/"table_d_modality.csv",table_d)
    write_csv(out/"table_e_reconciliation.csv",reconciliation); write_csv(out/"table_f_category.csv",categories)
    write_csv(out/"table_g_checks_per_cell.csv",mult); write_csv(out/"table_h_checks_by_paper.csv",per_paper)
    boot_payload={"method":"paper-clustered percentile bootstrap","papers":docs,"replicates":args.replicates,"seed":args.seed,"metrics":boot}
    (out/"bootstrap.json").write_text(json.dumps(boot_payload,indent=2)+"\n")
    plot(out/"evisearch_error_by_verifier_outcome.png",comp,boot)
    checked_counts=[x["check_count"] for x in checked]
    summary={"definitions":{"graded_score":"(correctness + completeness) / 2","fully_correct":"correctness == 1 and completeness == 1","error":"not fully_correct","not_reproduced_comparison":"all checked cells with recorded verified == false, including explicit partial checks"},
             "audit":audit,"coverage":{"cells":len(rows),"checked_cells":len(checked),"checked_rate":pct(len(checked),len(rows)),"not_checked_cells":len(rows)-len(checked),"not_checked_rate":pct(len(rows)-len(checked),len(rows)),"detailed_checks":len(checks),"mean_checks_per_checked_cell":round(mean(checked_counts),3),"median_checks_per_checked_cell":median(checked_counts),"max_checks_per_checked_cell":max(checked_counts)},
             "comparison":comp,"table_a":table_a,"table_b":table_b,"table_c":table_c,"table_d":table_d,"reconciliation":reconciliation,"categories":categories,"checks_per_cell":mult,"checks_by_paper":per_paper,"bootstrap":boot}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    cirows=[{"metric":k,**v} for k,v in boot.items() if k!="error_risk_ratio_not_reproduced_over_reproduced"]
    report=f"""# E3 — Verifier / Reproduction Reliability\n\nOffline analysis of {len(rows):,} scored EviSearch cells from {len(docs)} papers. No inference or rescoring was performed. Graded accuracy is `(correctness + completeness) / 2`; an error is any cell that is not fully correct. Associations are descriptive, not causal or calibrated-confidence claims.\n\n## Audit and coverage\n\n- Checked cells: {len(checked)}/{len(rows)} ({pct(len(checked),len(rows))}%)\n- Not checked: {len(rows)-len(checked)}/{len(rows)} ({pct(len(rows)-len(checked),len(rows))}%)\n- Verifier calls: {audit['verifier_calls']}; detailed checks: {len(checks)}; log files: {audit['log_files']}\n- Checks per checked cell: mean {mean(checked_counts):.3f}, median {median(checked_counts):g}, maximum {max(checked_counts)}\n- Multi-check cells: {audit['multi_check_cells']}\n- All detailed checks mapped unambiguously to `(paper, column)`. More checks than checked cells occur because a cell may have multiple evidence checks.\n\n## Table A — Accuracy by verifier state\n\n{md_table(table_a,[('verifier_state','State'),('count','N'),('percent','%'),('graded_accuracy','Graded accuracy'),('fully_correct_rate','Fully correct'),('error_count','Errors'),('error_rate','Error rate')])}\n\n## Reproduction as a risk signal among checked cells\n\n- Reproduced error rate: {100*comp['error_rate_reproduced']:.2f}%\n- Not-reproduced error rate: {100*comp['error_rate_not_reproduced']:.2f}%\n- Absolute difference (not reproduced minus reproduced): {100*comp['error_rate_difference_not_reproduced_minus_reproduced']:.2f} percentage points\n- Error risk ratio: {comp['error_risk_ratio_not_reproduced_over_reproduced']:.3f}\n- Graded-accuracy difference (reproduced minus not reproduced): {100*comp['graded_accuracy_difference_reproduced_minus_not_reproduced']:.2f} percentage points\n\n## Table B — Verifier outcome × review flag\n\n{md_table(table_b,[('verifier_group','Verifier'),('needs_review','Flagged'),('count','N'),('graded_accuracy','Graded accuracy'),('fully_correct_rate','Fully correct'),('error_count','Errors'),('error_rate','Error rate')])}\n\n## Table C — Verifier outcome × A/B agreement\n\n{md_table(table_c,[('agents_agree','A = B'),('verifier_group','Verifier'),('count','N'),('graded_accuracy','Graded accuracy'),('error_count','Errors'),('error_rate','Error rate')])}\n\n## Table D — Detailed-check modality\n\nModality rows can overlap at the cell level; statistics use unique cells within each modality, while check support uses detailed checks.\n\n{md_table(table_d,[('modality','Modality'),('detailed_checks','Checks'),('unique_cells','Unique cells'),('supported_checks','Supported checks'),('supported_check_rate','Support rate'),('graded_accuracy','Graded accuracy'),('error_count','Errors'),('error_rate','Error rate')])}\n\n## Paper-clustered bootstrap 95% CIs\n\nPercentile bootstrap with {args.replicates:,} paper-resampling replicates and seed `{args.seed}`. `valid_replicates` explicitly reports replicates containing the required subgroup.\n\n{md_table(cirows,[('metric','Metric'),('estimate','Estimate'),('ci_low','CI low'),('ci_high','CI high'),('valid_replicates','Valid replicates')])}\n\nAdditional reconciliation, scoring-category, checks-per-cell, and per-paper tables are provided as CSV files.\n"""
    (out/"report.md").write_text(report,encoding="utf-8")
    print(report); print(f"Outputs: {out}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
