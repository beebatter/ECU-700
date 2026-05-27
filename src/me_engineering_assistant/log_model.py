from __future__ import annotations

import argparse
import json
from pathlib import Path

from me_engineering_assistant.evaluate import evaluate_golden, evaluation_metrics, load_questions
from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.model import ECUAssistantPyFunc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log the ME ECU assistant as an MLflow pyfunc model.")
    parser.add_argument("--docs-dir", default=".", help="Directory containing ECU markdown source files.")
    parser.add_argument("--questions", default=None, help="Golden evaluation CSV.")
    parser.add_argument("--experiment-name", default="/Shared/me-engineering-assistant")
    parser.add_argument("--registered-model-name", default=None)
    parser.add_argument("--artifact-path", default="me_engineering_assistant")
    args = parser.parse_args(argv)

    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("mlflow is required to log the model. Install the databricks extra.") from exc

    docs_dir = Path(args.docs_dir).expanduser().resolve()
    src_dir = Path(__file__).resolve().parents[2]
    agent = ECUAgent(docs_dir=docs_dir)
    report = evaluate_golden(agent, load_questions(args.questions))

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="log-me-engineering-assistant"):
        mlflow.log_metrics(evaluation_metrics(report, prefix="golden_"))
        mlflow.log_text(json.dumps(report, indent=2), "evaluation/golden_report.json")
        model_info = mlflow.pyfunc.log_model(
            artifact_path=args.artifact_path,
            python_model=ECUAssistantPyFunc(),
            artifacts={"docs_dir": str(docs_dir)},
            code_paths=[str(src_dir)],
            registered_model_name=args.registered_model_name,
            input_example={"query": "How much RAM does the ECU-850 have?"},
        )
        print(json.dumps({"model_uri": model_info.model_uri, "evaluation": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
