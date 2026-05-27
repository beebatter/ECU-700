from __future__ import annotations

from pathlib import Path
from typing import Any

from me_engineering_assistant.graph import ECUAgent


try:
    import mlflow

    PythonModelBase = mlflow.pyfunc.PythonModel
except ImportError:
    PythonModelBase = object


class ECUAssistantPyFunc(PythonModelBase):
    def __init__(self, docs_dir: str | None = None) -> None:
        self.docs_dir = docs_dir
        self.agent: ECUAgent | None = None

    def load_context(self, context: Any) -> None:
        docs_dir = self.docs_dir
        if docs_dir is None and context is not None:
            artifacts = getattr(context, "artifacts", {}) or {}
            docs_dir = artifacts.get("docs_dir")
        self.agent = ECUAgent(docs_dir=docs_dir)

    def predict(self, context, model_input, params=None):
        if self.agent is None:
            self.load_context(context)

        queries, single = coerce_queries(model_input)
        responses = [self.agent.answer(query).to_dict() for query in queries]  # type: ignore[union-attr]
        return responses[0] if single else responses


def coerce_queries(model_input: Any) -> tuple[list[str], bool]:
    if isinstance(model_input, str):
        return [model_input], True

    if isinstance(model_input, dict):
        value = model_input.get("query") or model_input.get("Question") or model_input.get("question")
        if isinstance(value, str):
            return [value], True
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value], False

    if hasattr(model_input, "columns"):
        columns = list(getattr(model_input, "columns"))
        for column in ("query", "Question", "question"):
            if column in columns:
                values = model_input[column]
                if hasattr(values, "tolist"):
                    values = values.tolist()
                return [str(item) for item in values], False

    if isinstance(model_input, (list, tuple)):
        if not model_input:
            return [], False
        if all(isinstance(item, dict) for item in model_input):
            return [str(item.get("query") or item.get("Question") or item.get("question")) for item in model_input], False
        return [str(item) for item in model_input], False

    path = Path(str(model_input))
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], False

    return [str(model_input)], True
