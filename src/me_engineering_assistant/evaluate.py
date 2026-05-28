from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from me_engineering_assistant.documents import project_root
from me_engineering_assistant.graph import ECUAgent


EXPECTED_HITS = {
    "1": ("ECU-750", "+85"),
    "2": ("ECU-850", "2 GB", "LPDDR4"),
    "3": ("ECU-850b", "5 TOPS", "NPU"),
    "4": ("ECU-850b", "5 TOPS", "4 GB", "1.5 GHz", "2 GB", "1.2 GHz"),
    "5": ("ECU-750", "1 Mbps", "ECU-850", "2 Mbps", "Dual"),
    "6": ("ECU-850b", "1.7A", "550mA"),
    "7": ("ECU-850", "ECU-850b", "ECU-750", ("does not support", "not supported")),
    "8": ("ECU-750", "2 MB", "ECU-850", "16 GB", "ECU-850b", "32 GB"),
    "9": ("ECU-850", "ECU-850b", "+105", "ECU-750", "+85"),
    "10": ("me-driver-ctl --enable-npu --mode=performance",),
}

EXPECTED_SOURCES = {
    "1": ("ECU-700_Series_Manual.md",),
    "2": ("ECU-800_Series_Base.md",),
    "3": ("ECU-800_Series_Plus.md",),
    "4": ("ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"),
    "5": ("ECU-700_Series_Manual.md", "ECU-800_Series_Base.md"),
    "6": ("ECU-800_Series_Plus.md",),
    "7": ("ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"),
    "8": ("ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"),
    "9": ("ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"),
    "10": ("ECU-800_Series_Plus.md",),
}

EXPECTED_ROUTE_MODELS = {
    "1": ("ECU-750",),
    "2": ("ECU-850",),
    "3": ("ECU-850b",),
    "4": ("ECU-850", "ECU-850b"),
    "5": ("ECU-750", "ECU-850"),
    "6": ("ECU-850b",),
    "7": ("ECU-750", "ECU-850", "ECU-850b"),
    "8": ("ECU-750", "ECU-850", "ECU-850b"),
    "9": ("ECU-750", "ECU-850", "ECU-850b"),
    "10": ("ECU-850b",),
}


@dataclass(frozen=True)
class EvaluationRow:
    question_id: str
    question: str
    answer: str
    passed: bool
    source_passed: bool
    route_passed: bool
    latency_seconds: float
    confidence: float
    sources: list[str]
    route_models: list[str]
    route_mode: str
    needs_review: bool
    review_id: str | None


def load_questions(path: str | Path | None = None) -> list[dict[str, str]]:
    questions_path = Path(path) if path else project_root() / "test-questions.csv"
    with questions_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_golden(agent: ECUAgent, questions: Iterable[dict[str, str]]) -> dict[str, object]:
    rows: list[EvaluationRow] = []
    for row in questions:
        started = time.perf_counter()
        response = agent.answer(row["Question"])
        latency = time.perf_counter() - started
        passed = _contains_expected_hits(row["Question_ID"], response.answer)
        source_passed = _contains_expected_sources(row["Question_ID"], response.sources)
        route_passed = _contains_expected_route_models(row["Question_ID"], response.route.models)
        rows.append(
            EvaluationRow(
                question_id=row["Question_ID"],
                question=row["Question"],
                answer=response.answer,
                passed=passed,
                source_passed=source_passed,
                route_passed=route_passed,
                latency_seconds=latency,
                confidence=response.confidence,
                sources=response.sources,
                route_models=response.route.models,
                route_mode=response.route.mode,
                needs_review=response.needs_review,
                review_id=response.review_id,
            )
        )

    total = len(rows)
    passed_count = sum(1 for row in rows if row.passed)
    source_passed_count = sum(1 for row in rows if row.source_passed)
    route_passed_count = sum(1 for row in rows if row.route_passed)
    low_confidence_count = sum(1 for row in rows if row.confidence < 0.75)
    needs_review_count = sum(1 for row in rows if row.needs_review)
    latencies = [row.latency_seconds for row in rows]
    average_latency = sum(row.latency_seconds for row in rows) / total if total else 0.0
    average_confidence = sum(row.confidence for row in rows) / total if total else 0.0
    return {
        "total": total,
        "passed": passed_count,
        "accuracy": passed_count / total if total else 0.0,
        "source_match_rate": source_passed_count / total if total else 0.0,
        "route_match_rate": route_passed_count / total if total else 0.0,
        "average_latency_seconds": average_latency,
        "p95_latency_seconds": percentile(latencies, 95),
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "average_confidence": average_confidence,
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": low_confidence_count / total if total else 0.0,
        "needs_review_count": needs_review_count,
        "needs_review_rate": needs_review_count / total if total else 0.0,
        "rows": [row.__dict__ for row in rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ECU assistant on the golden question set.")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--experiment-name", default="me-engineering-assistant-local")
    parser.add_argument("--run-name", default="golden-evaluation")
    args = parser.parse_args(argv)

    agent = ECUAgent(docs_dir=args.docs_dir)
    report = evaluate_golden(agent, load_questions(args.questions))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.log_mlflow:
        log_report_to_mlflow(report, experiment_name=args.experiment_name, run_name=args.run_name)
    print(json.dumps(report, indent=2))
    return 0


def _contains_expected_hits(question_id: str, answer: str) -> bool:
    expected = EXPECTED_HITS.get(question_id, ())
    answer_lower = answer.lower()
    return all(_contains_hit(answer_lower, hit) for hit in expected)


def _contains_hit(answer_lower: str, hit: str | tuple[str, ...]) -> bool:
    if isinstance(hit, tuple):
        return any(option.lower() in answer_lower for option in hit)
    return hit.lower() in answer_lower


def _contains_expected_sources(question_id: str, sources: list[str]) -> bool:
    expected = set(EXPECTED_SOURCES.get(question_id, ()))
    return expected.issubset(set(sources))


def _contains_expected_route_models(question_id: str, models: list[str]) -> bool:
    expected = set(EXPECTED_ROUTE_MODELS.get(question_id, ()))
    return expected.issubset(set(models))


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percent / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def evaluation_metrics(report: dict[str, object], prefix: str = "") -> dict[str, float]:
    excluded = {"rows"}
    metrics: dict[str, float] = {}
    for key, value in report.items():
        if key in excluded:
            continue
        if isinstance(value, bool):
            metrics[f"{prefix}{key}"] = float(value)
        elif isinstance(value, (int, float)):
            metrics[f"{prefix}{key}"] = float(value)
    return metrics


def log_report_to_mlflow(report: dict[str, object], experiment_name: str, run_name: str) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("mlflow is required for --log-mlflow. Install mlflow or .[eval].") from exc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(evaluation_metrics(report, prefix="golden_"))
        mlflow.log_text(json.dumps(report, indent=2), "evaluation/golden_report.json")


if __name__ == "__main__":
    raise SystemExit(main())
