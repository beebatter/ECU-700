from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class QAResult:
    question_id: str
    category: str
    question: str
    expected_answer: str
    evaluation_criteria: str
    actual_answer: str
    sources: list[str]
    confidence: float | None
    needs_review: bool | None
    passed: bool
    score: float
    latency_seconds: float
    reason: str


@dataclass
class QAEvalReport:
    total: int
    passed: int
    failed: int
    pass_rate: float
    average_score: float
    average_latency_seconds: float
    results: list[QAResult]


def find_project_root(start: Path) -> Path:
    current = start.resolve()

    for parent in [current, *current.parents]:
        if (
            (parent / "pyproject.toml").exists()
            or (parent / "pytest.ini").exists()
            or (parent / "setup.py").exists()
            or (parent / "tests").exists()
        ):
            return parent

    return start.resolve().parents[2]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.+°@/_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(expected_answer: str) -> list[str]:
    """
    Extract simple evaluation keywords from the expected answer.

    This is intentionally lightweight:
    - keeps technical numbers such as 85, 5 TOPS, 2 GB, 1.5 GHz
    - keeps ECU model names
    - removes very common stopwords
    """
    normalized = normalize_text(expected_answer)

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "for", "of",
        "and", "or", "in", "on", "with", "it", "this", "that", "has",
        "have", "features", "capable", "capability", "capabilities",
        "while", "from", "as", "by", "be", "up", "than", "per",
    }

    tokens = normalized.split()
    keywords: list[str] = []

    for token in tokens:
        if token in stopwords:
            continue
        if len(token) <= 2 and not any(ch.isdigit() for ch in token):
            continue
        keywords.append(token)

    seen = set()
    deduped = []
    for keyword in keywords:
        if keyword not in seen:
            deduped.append(keyword)
            seen.add(keyword)

    return deduped


def score_answer(expected_answer: str, actual_answer: str) -> tuple[float, bool, str]:
    expected_keywords = extract_keywords(expected_answer)
    actual_normalized = normalize_text(actual_answer)

    if not expected_keywords:
        return 0.0, False, "No evaluable keywords extracted from expected answer."

    matched = [kw for kw in expected_keywords if kw in actual_normalized]
    score = len(matched) / len(expected_keywords)

    passed = score >= 0.45

    reason = (
        f"Matched {len(matched)}/{len(expected_keywords)} expected keywords. "
        f"Matched: {', '.join(matched[:12])}"
    )

    return score, passed, reason


def load_questions(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "Question_ID",
            "Category",
            "Question",
            "Expected_Answer",
            "Evaluation_Criteria",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        return [dict(row) for row in reader]


def get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def run_agent_question(agent: Any, question: str) -> Any:
    """
    Supports both:
    - agent.answer(question)
    - agent.predict(question)
    """
    if hasattr(agent, "answer"):
        return agent.answer(question)
    if hasattr(agent, "predict"):
        return agent.predict(question)
    raise AttributeError("Agent object must expose answer() or predict().")


def build_agent(project_root: Path) -> Any:
    """
    Imports and constructs your ECUAgent.

    If your constructor needs different arguments, adjust this function.
    """
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    from me_engineering_assistant.graph import ECUAgent

    return ECUAgent()


def evaluate_questions(csv_path: Path, output_dir: Path, project_root: Path) -> QAEvalReport:
    questions = load_questions(csv_path)
    agent = build_agent(project_root)

    results: list[QAResult] = []

    for row in questions:
        question = row["Question"]
        expected_answer = row["Expected_Answer"]

        started = time.perf_counter()
        response = run_agent_question(agent, question)
        latency = time.perf_counter() - started

        actual_answer = str(
            get_attr_or_key(response, "answer", response)
        )

        sources_raw = get_attr_or_key(response, "sources", []) or []
        sources = [str(source) for source in sources_raw]

        confidence = get_attr_or_key(response, "confidence", None)
        needs_review = get_attr_or_key(response, "needs_review", None)

        score, passed, reason = score_answer(expected_answer, actual_answer)

        results.append(
            QAResult(
                question_id=row["Question_ID"],
                category=row["Category"],
                question=question,
                expected_answer=expected_answer,
                evaluation_criteria=row["Evaluation_Criteria"],
                actual_answer=actual_answer,
                sources=sources,
                confidence=confidence,
                needs_review=needs_review,
                passed=passed,
                score=score,
                latency_seconds=latency,
                reason=reason,
            )
        )

    total = len(results)
    passed_count = sum(1 for item in results if item.passed)
    failed_count = total - passed_count
    average_score = sum(item.score for item in results) / total if total else 0.0
    average_latency = (
        sum(item.latency_seconds for item in results) / total if total else 0.0
    )

    return QAEvalReport(
        total=total,
        passed=passed_count,
        failed=failed_count,
        pass_rate=passed_count / total if total else 0.0,
        average_score=average_score,
        average_latency_seconds=average_latency,
        results=results,
    )


def write_markdown_report(report: QAEvalReport, output_path: Path) -> None:
    lines: list[str] = []

    lines.append("# QA Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total questions: {report.total}")
    lines.append(f"- Passed: {report.passed}")
    lines.append(f"- Failed: {report.failed}")
    lines.append(f"- Pass rate: {report.pass_rate:.2%}")
    lines.append(f"- Average keyword score: {report.average_score:.2%}")
    lines.append(f"- Average latency: {report.average_latency_seconds:.2f}s")
    lines.append("")

    lines.append("## Dataset Coverage")
    lines.append("")
    category_counts: dict[str, int] = {}
    for result in report.results:
        category_counts[result.category] = category_counts.get(result.category, 0) + 1

    for category, count in sorted(category_counts.items()):
        lines.append(f"- {category}: {count}")
    lines.append("")

    lines.append("## Per-question Results")
    lines.append("")
    lines.append("| ID | Status | Score | Category | Latency | Question |")
    lines.append("|---:|---|---:|---|---:|---|")

    for item in report.results:
        status = "PASS" if item.passed else "FAIL"
        safe_question = item.question.replace("|", "\\|")
        safe_category = item.category.replace("|", "\\|")
        lines.append(
            f"| {item.question_id} | {status} | {item.score:.0%} | "
            f"{safe_category} | {item.latency_seconds:.2f}s | {safe_question} |"
        )

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for item in report.results:
        status = "PASS" if item.passed else "FAIL"
        lines.append(f"### Q{item.question_id}: {status}")
        lines.append("")
        lines.append(f"**Category:** {item.category}")
        lines.append("")
        lines.append(f"**Question:** {item.question}")
        lines.append("")
        lines.append("**Expected answer:**")
        lines.append("")
        lines.append(item.expected_answer)
        lines.append("")
        lines.append("**Actual answer:**")
        lines.append("")
        lines.append(item.actual_answer)
        lines.append("")
        lines.append(f"**Score:** {item.score:.2%}")
        lines.append("")
        lines.append(f"**Latency:** {item.latency_seconds:.2f}s")
        lines.append("")
        lines.append(f"**Confidence:** {item.confidence}")
        lines.append("")
        lines.append(f"**Needs review:** {item.needs_review}")
        lines.append("")
        lines.append("**Sources:**")
        lines.append("")
        if item.sources:
            for source in item.sources:
                lines.append(f"- {source}")
        else:
            lines.append("- No sources returned")
        lines.append("")
        lines.append("**Evaluation reason:**")
        lines.append("")
        lines.append(item.reason)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ECU assistant against a QA dataset and generate an evaluation report."
    )
    parser.add_argument(
        "--questions",
        default="test-questions.csv",
        help="Path to QA test dataset CSV. Default: test-questions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="qa_eval_reports",
        help="Directory for generated QA evaluation reports. Default: qa_eval_reports",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).parent)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.questions)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    if not csv_path.exists():
        print(f"QA dataset not found: {csv_path}", file=sys.stderr)
        return 1

    report = evaluate_questions(
        csv_path=csv_path,
        output_dir=output_dir,
        project_root=project_root,
    )

    json_path = output_dir / "qa_eval_report.json"
    markdown_path = output_dir / "qa_eval_report.md"

    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(report, markdown_path)

    print(f"QA JSON report: {json_path}")
    print(f"QA Markdown report: {markdown_path}")

    if report.failed > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
