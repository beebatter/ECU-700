from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class TestCaseResult:
    classname: str
    name: str
    time: float
    status: str
    message: str = ""


@dataclass
class TestReport:
    command: list[str]
    returncode: int
    duration_seconds: float
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    cases: list[TestCaseResult]


def find_project_root(start: Path) -> Path:
    """Find the real project root from this script's location.

    This script may live in either:
    - project_root/scripts/generate_test_report.py
    - project_root/src/me_engineering_assistant/generate_test_report.py

    We walk upwards until we find common project markers. Preference is given
    to directories that contain a tests folder or pyproject.toml.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    markers = ("pyproject.toml", "pytest.ini", "setup.py", "setup.cfg")
    for parent in (current, *current.parents):
        if (parent / "tests").exists() or any((parent / marker).exists() for marker in markers):
            return parent

    # Fallback for src/me_engineering_assistant/generate_test_report.py
    # -> project_root is three levels above the file.
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return current


def build_pytest_command(test_paths: Iterable[str], junit_path: Path, extra_pytest_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        *test_paths,
        "--junitxml",
        str(junit_path),
        "-q",
        *extra_pytest_args,
    ]


def run_tests(
    project_root: Path,
    junit_path: Path,
    test_paths: list[str],
    extra_pytest_args: list[str],
) -> tuple[list[str], int, float]:
    command = build_pytest_command(test_paths, junit_path, extra_pytest_args)

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.perf_counter() - started

    console_log = junit_path.with_suffix(".console.txt")
    console_log.write_text(completed.stdout, encoding="utf-8")

    return command, completed.returncode, duration


def _read_int(element: ET.Element, key: str) -> int:
    return int(element.attrib.get(key, "0") or 0)


def parse_junit(junit_path: Path, command: list[str], returncode: int, duration: float) -> TestReport:
    tree = ET.parse(junit_path)
    root = tree.getroot()

    cases: list[TestCaseResult] = []

    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.iter("testsuite"))

    total = sum(_read_int(suite, "tests") for suite in suites)
    failed = sum(_read_int(suite, "failures") for suite in suites)
    errors = sum(_read_int(suite, "errors") for suite in suites)
    skipped = sum(_read_int(suite, "skipped") for suite in suites)

    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        case_time = float(case.attrib.get("time", "0") or 0)

        status = "passed"
        message = ""

        failure = case.find("failure")
        error = case.find("error")
        skip = case.find("skipped")

        if failure is not None:
            status = "failed"
            message = failure.attrib.get("message", "") or (failure.text or "")
        elif error is not None:
            status = "error"
            message = error.attrib.get("message", "") or (error.text or "")
        elif skip is not None:
            status = "skipped"
            message = skip.attrib.get("message", "") or (skip.text or "")

        cases.append(
            TestCaseResult(
                classname=classname,
                name=name,
                time=case_time,
                status=status,
                message=message.strip(),
            )
        )

    passed = total - failed - errors - skipped

    return TestReport(
        command=command,
        returncode=returncode,
        duration_seconds=duration,
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        cases=cases,
    )


def write_markdown_report(report: TestReport, output_path: Path) -> None:
    pass_rate = (report.passed / report.total * 100) if report.total else 0.0

    lines: list[str] = []
    lines.append("# Test Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total: {report.total}")
    lines.append(f"- Passed: {report.passed}")
    lines.append(f"- Failed: {report.failed}")
    lines.append(f"- Errors: {report.errors}")
    lines.append(f"- Skipped: {report.skipped}")
    lines.append(f"- Pass rate: {pass_rate:.2f}%")
    lines.append(f"- Duration: {report.duration_seconds:.2f}s")
    lines.append(f"- Return code: {report.returncode}")
    lines.append("")
    lines.append("## Command")
    lines.append("")
    lines.append("```bash")
    lines.append(" ".join(report.command))
    lines.append("```")
    lines.append("")

    lines.append("## Test Cases")
    lines.append("")
    lines.append("| Status | Test | Time |")
    lines.append("|---|---|---:|")

    for case in report.cases:
        test_name = f"{case.classname}.{case.name}".strip(".")
        lines.append(f"| {case.status} | `{test_name}` | {case.time:.3f}s |")

    problem_cases = [case for case in report.cases if case.status in {"failed", "error"}]

    if problem_cases:
        lines.append("")
        lines.append("## Failures and Errors")
        lines.append("")

        for case in problem_cases:
            test_name = f"{case.classname}.{case.name}".strip(".")
            lines.append(f"### {test_name}")
            lines.append("")
            lines.append(f"- Status: {case.status}")
            lines.append("")
            lines.append("```text")
            lines.append(case.message[:4000])
            lines.append("```")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest and generate test reports.")
    parser.add_argument(
        "--test-path",
        action="append",
        default=None,
        help="Test path passed to pytest. Can be used multiple times. Default: tests if it exists, otherwise current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_reports",
        help="Directory for generated reports, relative to project root unless absolute. Default: test_reports",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to pytest after '--'. Example: -- -k answering -vv",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())

    test_paths = args.test_path
    if not test_paths:
        test_paths = ["tests"] if (project_root / "tests").exists() else ["."]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    junit_path = output_dir / "junit.xml"
    json_path = output_dir / "test_report.json"
    markdown_path = output_dir / "test_report.md"

    extra_pytest_args = args.pytest_args
    if extra_pytest_args and extra_pytest_args[0] == "--":
        extra_pytest_args = extra_pytest_args[1:]

    command, returncode, duration = run_tests(project_root, junit_path, test_paths, extra_pytest_args)

    if not junit_path.exists():
        print(f"JUnit report was not created: {junit_path}", file=sys.stderr)
        print(f"Console log: {junit_path.with_suffix('.console.txt')}", file=sys.stderr)
        return returncode or 1

    report = parse_junit(junit_path, command, returncode, duration)

    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, markdown_path)

    print(f"Project root: {project_root}")
    print(f"JUnit XML: {junit_path}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Console log: {junit_path.with_suffix('.console.txt')}")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
