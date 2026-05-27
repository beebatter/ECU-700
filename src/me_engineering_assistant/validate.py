from __future__ import annotations

import argparse
import json

from me_engineering_assistant.evaluate import evaluate_golden, load_questions
from me_engineering_assistant.graph import ECUAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the ECU assistant before model logging.")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    args = parser.parse_args(argv)

    agent = ECUAgent(docs_dir=args.docs_dir)
    report = evaluate_golden(agent, load_questions(args.questions))
    print(json.dumps(report, indent=2))
    if report["accuracy"] < args.min_accuracy:
        raise SystemExit(f"Validation failed: accuracy {report['accuracy']:.2%} < {args.min_accuracy:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

