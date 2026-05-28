from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from me_engineering_assistant.graph import ECUAgent


try:
    import mlflow

    PythonModelBase = mlflow.pyfunc.PythonModel
except ImportError:
    PythonModelBase = object


class ECUAssistantPyFunc(PythonModelBase):
    def __init__(
        self,
        docs_dir: str | None = None,
        *,
        memory_enabled: bool | None = None,
        memory_path: str | None = None,
        memory_scope: str = "global",
    ) -> None:
        self.docs_dir = docs_dir
        self.memory_enabled = memory_enabled
        self.memory_path = memory_path
        self.memory_scope = memory_scope
        self.agent: ECUAgent | None = None

    def load_context(self, context: Any) -> None:
        docs_dir = self.docs_dir
        if docs_dir is None and context is not None:
            artifacts = getattr(context, "artifacts", {}) or {}
            docs_dir = artifacts.get("docs_dir")
        self.agent = ECUAgent(
            docs_dir=docs_dir,
            memory_enabled=self.memory_enabled,
            memory_path=self.memory_path,
            memory_scope=self.memory_scope,
        )

    def predict(self, context, model_input, params=None):
        if self.agent is None:
            self.load_context(context)

        requests, single = coerce_prediction_requests(model_input)
        responses = [
            self.agent.answer(request.query, session_id=request.session_id).to_dict()  # type: ignore[union-attr]
            for request in requests
        ]
        return responses[0] if single else responses


@dataclass(frozen=True)
class PredictionRequest:
    query: str
    session_id: str | None = None


def coerce_prediction_requests(model_input: Any) -> tuple[list[PredictionRequest], bool]:
    if isinstance(model_input, dict):
        value = model_input.get("query") or model_input.get("Question") or model_input.get("question")
        session_value = model_input.get("session_id") or model_input.get("session") or model_input.get("conversation_id")
        if isinstance(value, str):
            return [PredictionRequest(query=value, session_id=_optional_string(session_value))], True
        if isinstance(value, (list, tuple)):
            sessions = _align_sessions(session_value, len(value))
            return [
                PredictionRequest(query=str(item), session_id=sessions[index])
                for index, item in enumerate(value)
            ], False

    if hasattr(model_input, "columns"):
        columns = list(getattr(model_input, "columns"))
        for column in ("query", "Question", "question"):
            if column in columns:
                values = model_input[column]
                if hasattr(values, "tolist"):
                    values = values.tolist()
                sessions = _dataframe_sessions(model_input, len(values))
                return [
                    PredictionRequest(query=str(item), session_id=sessions[index])
                    for index, item in enumerate(values)
                ], False

    if isinstance(model_input, (list, tuple)) and model_input and all(isinstance(item, dict) for item in model_input):
        return [
            PredictionRequest(
                query=str(item.get("query") or item.get("Question") or item.get("question")),
                session_id=_optional_string(item.get("session_id") or item.get("session") or item.get("conversation_id")),
            )
            for item in model_input
        ], False

    queries, single = coerce_queries(model_input)
    return [PredictionRequest(query=query) for query in queries], single


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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _align_sessions(value: Any, length: int) -> list[str | None]:
    if isinstance(value, (list, tuple)):
        sessions = [_optional_string(item) for item in value]
        return sessions + [None] * max(0, length - len(sessions))
    return [_optional_string(value)] * length


def _dataframe_sessions(model_input: Any, length: int) -> list[str | None]:
    columns = list(getattr(model_input, "columns"))
    for column in ("session_id", "session", "conversation_id"):
        if column in columns:
            values = model_input[column]
            if hasattr(values, "tolist"):
                values = values.tolist()
            return _align_sessions(values, length)
    return [None] * length
