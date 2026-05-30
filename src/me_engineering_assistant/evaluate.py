from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from me_engineering_assistant.documents import project_root
from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.planner import models_in_query


PASS_SCORE_THRESHOLD = 0.45


@dataclass(frozen=True)
class EvaluationRow:
    question_id: str
    question: str
    expected_answer: str
    answer: str
    passed: bool
    score: float
    matched_keywords: list[str]
    source_passed: bool
    route_passed: bool
    latency_seconds: float
    llm_calls: int
    llm_latency_seconds: float
    retrieval_latency_seconds: float
    generation_latency_seconds: float
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
        score, matched_keywords = score_answer(row.get("Expected_Answer", ""), response.answer)
        expected_models = expected_models_for_row(row, agent)
        source_passed = expected_sources_match(expected_models, response.sources, agent)
        route_passed = expected_route_match(expected_models, response.route.models)
        passed = score >= PASS_SCORE_THRESHOLD and source_passed
        rows.append(
            EvaluationRow(
                question_id=row["Question_ID"],
                question=row["Question"],
                expected_answer=row.get("Expected_Answer", ""),
                answer=response.answer,
                passed=passed,
                score=score,
                matched_keywords=matched_keywords,
                source_passed=source_passed,
                route_passed=route_passed,
                latency_seconds=latency,
                llm_calls=response.llm_calls,
                llm_latency_seconds=response.llm_latency_seconds,
                retrieval_latency_seconds=response.retrieval_latency_seconds,
                generation_latency_seconds=response.generation_latency_seconds,
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
    average_score = sum(row.score for row in rows) / total if total else 0.0
    average_llm_calls = sum(row.llm_calls for row in rows) / total if total else 0.0
    average_llm_latency = sum(row.llm_latency_seconds for row in rows) / total if total else 0.0
    average_retrieval_latency = sum(row.retrieval_latency_seconds for row in rows) / total if total else 0.0
    average_generation_latency = sum(row.generation_latency_seconds for row in rows) / total if total else 0.0
    return {
        "total": total,
        "passed": passed_count,
        "accuracy": passed_count / total if total else 0.0,
        "average_score": average_score,
        "source_match_rate": source_passed_count / total if total else 0.0,
        "route_match_rate": route_passed_count / total if total else 0.0,
        "average_latency_seconds": average_latency,
        "p95_latency_seconds": percentile(latencies, 95),
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "average_llm_calls": average_llm_calls,
        "average_llm_latency_seconds": average_llm_latency,
        "average_retrieval_latency_seconds": average_retrieval_latency,
        "average_generation_latency_seconds": average_generation_latency,
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


def score_answer(expected_answer: str, answer: str) -> tuple[float, list[str]]:
    keywords = extract_keywords(expected_answer)
    if not keywords:
        return 0.0, []
    normalized_answer = normalize_text(answer)
    matched = [keyword for keyword in keywords if keyword in normalized_answer]
    return len(matched) / len(keywords), matched


def extract_keywords(expected_answer: str) -> list[str]:
    normalized = normalize_text(expected_answer)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "and",
        "or", "in", "on", "with", "it", "this", "that", "has", "have", "features",
        "capable", "capability", "capabilities", "while", "from", "as", "by", "be",
        "up", "than", "per", "series", "model", "models", "makes", "suitable",
    }
    keywords = []
    for token in normalized.split():
        if token in stopwords:
            continue
        if len(token) <= 2 and not any(character.isdigit() for character in token):
            continue
        keywords.append(token)
    return _dedupe(keywords)


def normalize_text(value: str) -> str:
    text = value.lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[^a-z0-9.+°@/_=-]+", " ", text)
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s+(gb|kb|mb|mbps|tops|ghz|mhz|ma|a|v|°c|c)\b",
        r"\1\2",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expected_models_for_row(row: dict[str, str], agent: ECUAgent) -> list[str]:
    text = f"{row.get('Question', '')} {row.get('Expected_Answer', '')}"
    return models_in_query(text, catalog=agent.catalog, include_unknown=False)


def expected_sources_match(expected_models: list[str], sources: list[str], agent: ECUAgent) -> bool:
    if not expected_models:
        return bool(sources)
    expected_sources = {
        entry.source
        for entry in agent.catalog
        if entry.model in set(expected_models)
    }
    return expected_sources.issubset(set(sources))


def expected_route_match(expected_models: list[str], route_models: list[str]) -> bool:
    if not expected_models:
        return True
    return set(expected_models).issubset(set(route_models))


def _dedupe(values: list[str]) -> list[str]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


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
