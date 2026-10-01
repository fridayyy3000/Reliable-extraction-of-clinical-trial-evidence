#!/usr/bin/env python3
"""Score blinded E1 outputs with a fixed Vertex Gemini judge.

This is deliberately a thin transport layer around ``src.evaluation.claude_scoring``.  That module remains the
authority for cell normalization, queue construction, label ingestion, and metric aggregation; this script only
sends its blinded queue batches to the repository's existing Vertex adapter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import MAX_TOKENS  # noqa: E402
from src.evaluation import claude_scoring  # noqa: E402
from src.inference import InferenceError, Message, Usage, get_chat  # noqa: E402


class Judgment(BaseModel):
    id: str
    correctness: float
    completeness: float
    reason: str = Field(min_length=1)


class BatchJudgment(BaseModel):
    batch: str
    results: List[Judgment]


def response_schema(batch: Dict[str, Any]) -> Dict[str, Any]:
    """The constrained response accepted by ``claude_scoring.ingest``."""
    return {
        "type": "object",
        "properties": {
            "batch": {"type": "string"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "correctness": {"type": "number"},
                        "completeness": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "correctness", "completeness", "reason"],
                },
            },
        },
        "required": ["batch", "results"],
    }


def prompt(rubric: str, batch: Dict[str, Any]) -> str:
    """The checked-in rubric verbatim, followed only by the blinded batch it asks the judge to score."""
    return rubric.rstrip() + "\n\n## Blinded batch\n\n```json\n" + json.dumps(batch, indent=2, ensure_ascii=False) + "\n```\n"


def validate_payload(payload: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed before the existing ingester appends anything to the label store."""
    parsed = BatchJudgment.model_validate(payload)
    result = parsed.model_dump()
    expected = [item["id"] for item in batch["items"]]
    returned = [item["id"] for item in result["results"]]
    if result["batch"] != batch["batch"]:
        raise ValueError(f"returned batch {result['batch']!r}, expected {batch['batch']!r}")
    if len(returned) != len(set(returned)):
        raise ValueError("response contains duplicate item ids")
    if set(returned) != set(expected):
        raise ValueError(f"response ids differ from queue (missing={set(expected) - set(returned)}, "
                         f"unknown={set(returned) - set(expected)})")
    allowed = set(claude_scoring.SCORES)
    for item in result["results"]:
        if item["correctness"] not in allowed or item["completeness"] not in allowed:
            raise ValueError(f"{item['id']}: score outside {sorted(allowed)}")
        item["reason"] = item["reason"].strip()
        if not item["reason"]:
            raise ValueError(f"{item['id']}: empty reason")
    result["results"].sort(key=lambda item: expected.index(item["id"]))
    return result


def score_batch(chat: Any, rubric: str, queue_path: Path, result_dir: Path, temperature: float,
                max_retries: int) -> tuple[Path, Usage]:
    batch = json.loads(queue_path.read_text(encoding="utf-8"))
    schema = response_schema(batch)
    request = prompt(rubric, batch)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            reply = chat.chat(
                [Message.user(request)],
                response_schema=schema,
                temperature=temperature,
                max_tokens=MAX_TOKENS["judge"],
            )
            payload = validate_payload(reply.json(), batch)
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / queue_path.name
            result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return result_path, reply.usage
        except (InferenceError, ValueError, ValidationError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"{queue_path.name}: judge failed after {max_retries} attempts: {last_error}")


def all_cells(systems: Sequence[claude_scoring.System], docs: Sequence[str]) -> List[claude_scoring.Cell]:
    columns = claude_scoring.load_columns()
    gold = claude_scoring.load_gold()
    return [
        cell
        for system in systems
        for cell in claude_scoring.cells(system, docs, columns=columns, gold=gold)
    ]


def plan(systems: Sequence[claude_scoring.System], docs: Sequence[str], labels: Dict[str, dict]) -> Dict[str, Any]:
    cells = all_cells(systems, docs)
    pending = {cell.id for cell in cells if not cell.mechanical and cell.id not in labels}
    return {
        "systems": [system.name for system in systems],
        "documents": len(docs),
        "cells": len(cells),
        "mechanical": sum(cell.mechanical for cell in cells),
        "already_labeled": sum(not cell.mechanical and cell.id in labels for cell in cells),
        "unique_pending_judgments": len(pending),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Fixed catalog model key used as judge")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Reserved for interface stability; scoring is serialized for deterministic ingestion")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--docs", default="all")
    parser.add_argument("--scorer", required=True, help="Immutable judge/config identifier written to labels")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--system", action="append", required=True,
                        help="name=run/stage; repeat for every system")
    parser.add_argument("--plan", action="store_true", help="Print counts without writing files or calling Vertex")
    args = parser.parse_args(argv)
    if args.temperature != 0.0:
        parser.error("this fixed-judge runner requires --temperature 0")
    if args.workers != 1:
        parser.error("this fixed-judge runner requires --workers 1 for deterministic label ingestion")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    systems = [claude_scoring.System.parse(spec) for spec in args.system]
    gold = claude_scoring.load_gold()
    docs = claude_scoring.select_docs(args.docs, gold)
    labels = claude_scoring.load_labels(args.labels)
    run_plan = plan(systems, docs, labels)
    print(json.dumps({"model": args.model, "temperature": args.temperature, "scorer": args.scorer, **run_plan}, indent=2))
    if args.plan:
        return 0

    rubric_path = PROJECT_ROOT / "experiment-scripts" / "scoring" / "RUBRIC.md"
    rubric = rubric_path.read_text(encoding="utf-8")
    queue_paths = claude_scoring.build_queue(all_cells(systems, docs), labels, args.queue_dir)
    print(f"[scoring] queued {len(queue_paths)} blinded batches")
    chat = get_chat("judge", args.model)
    usage = Usage()
    for queue_path in tqdm(queue_paths, desc="Scoring", unit="batch"):
        result_path, batch_usage = score_batch(
            chat, rubric, queue_path, args.result_dir, args.temperature, args.max_retries
        )
        claude_scoring.ingest(result_path, args.queue_dir, args.labels, args.scorer)
        usage.add(batch_usage)

    final_labels = claude_scoring.load_labels(args.labels)
    reports = [claude_scoring.report(system, docs, final_labels) for system in systems]
    output = {
        "judge": {
            "provider": "vertex",
            "model": args.model,
            "temperature": args.temperature,
            "scorer": args.scorer,
            "rubric": str(rubric_path.relative_to(PROJECT_ROOT)),
            "labels": str(args.labels),
            "usage": usage.to_dict(),
        },
        "documents": docs,
        "reports": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    incomplete = [report["system"] for report in reports if not report["complete"]]
    print(f"[scoring] report: {args.report}")
    for report in reports:
        print(f"[scoring] {report['system']}: {report['overall']} unscored={report['unscored']}")
    if incomplete:
        print(f"[scoring] incomplete systems: {', '.join(incomplete)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
