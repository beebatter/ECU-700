from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from me_engineering_assistant.knowledge import ECUSpec
from me_engineering_assistant.retriever import InMemoryECURetriever, RetrievalResult


SPEC_FIELDS = (
    "processor",
    "memory_ram",
    "storage",
    "can_interface",
    "ethernet",
    "power_consumption",
    "operating_temperature",
    "connectors",
    "npu",
    "ota_supported",
    "npu_enable_command",
    "safety",
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    result: Any
    sources: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ECUToolbox:
    def __init__(self, specs: Mapping[str, ECUSpec], retriever: InMemoryECURetriever) -> None:
        self.specs = dict(specs)
        self.retriever = retriever

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_documents",
                "description": "Search ECU markdown documentation chunks for evidence.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "models": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_model_spec",
                "description": "Read extracted specifications for one ECU model.",
                "input_schema": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"],
                },
            },
            {
                "name": "compare_model_specs",
                "description": "Compare selected specification fields across ECU models.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "models": {"type": "array", "items": {"type": "string"}},
                        "fields": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["models"],
                },
            },
            {
                "name": "list_models",
                "description": "List ECU models currently available in the internal documentation.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name == "search_documents":
            return self.search_documents(**call.arguments)
        if call.name == "read_model_spec":
            return self.read_model_spec(**call.arguments)
        if call.name == "compare_model_specs":
            return self.compare_model_specs(**call.arguments)
        if call.name == "list_models":
            return self.list_models()
        return ToolResult(
            name=call.name,
            arguments=call.arguments,
            result={"error": f"Unknown tool: {call.name}"},
            sources=[],
        )

    def search_documents(
        self,
        query: str,
        models: Sequence[str] | None = None,
        top_k: int = 4,
    ) -> ToolResult:
        filters = {"models": list(models)} if models else None
        results = self.retriever.retrieve(query, filters=filters, top_k=top_k)
        payload = [
            {
                "content": result.content,
                "metadata": result.metadata,
                "score": result.score,
            }
            for result in results
        ]
        return ToolResult(
            name="search_documents",
            arguments={"query": query, "models": list(models or ()), "top_k": top_k},
            result=payload,
            sources=_unique_sources(result.metadata.get("source") for result in results),
        )

    def read_model_spec(self, model: str) -> ToolResult:
        spec = self.specs.get(normalize_model_name(model))
        if spec is None:
            return ToolResult(
                name="read_model_spec",
                arguments={"model": model},
                result={"error": f"No specification found for {model}"},
                sources=[],
            )
        return ToolResult(
            name="read_model_spec",
            arguments={"model": spec.model},
            result=spec_to_dict(spec),
            sources=[spec.source],
        )

    def compare_model_specs(
        self,
        models: Sequence[str],
        fields: Sequence[str] | None = None,
    ) -> ToolResult:
        normalized_models = [normalize_model_name(model) for model in models]
        selected_fields = normalize_fields(fields or SPEC_FIELDS)
        rows: list[dict[str, Any]] = []
        sources: list[str] = []
        for model in normalized_models:
            spec = self.specs.get(model)
            if spec is None:
                rows.append({"model": model, "error": "No specification found"})
                continue
            spec_dict = spec_to_dict(spec)
            row = {"model": spec.model, "series": spec.series, "source": spec.source}
            for field in selected_fields:
                row[field] = spec_dict.get(field)
            rows.append(row)
            sources.append(spec.source)
        return ToolResult(
            name="compare_model_specs",
            arguments={"models": normalized_models, "fields": selected_fields},
            result=rows,
            sources=_unique_sources(sources),
        )

    def list_models(self) -> ToolResult:
        rows = [
            {"model": spec.model, "series": spec.series, "source": spec.source}
            for spec in self.specs.values()
        ]
        return ToolResult(
            name="list_models",
            arguments={},
            result=rows,
            sources=_unique_sources(spec.source for spec in self.specs.values()),
        )


def normalize_model_name(model: str) -> str:
    lowered = model.lower().replace(" ", "-")
    if "750" in lowered or "700" in lowered:
        return "ECU-750"
    if "850b" in lowered or "800b" in lowered:
        return "ECU-850b"
    if "850" in lowered or "800a" in lowered or "800" in lowered:
        return "ECU-850"
    return model


def normalize_fields(fields: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for field in fields:
        clean = field.lower().strip().replace(" ", "_").replace(".", "")
        aliases = {
            "ram": "memory_ram",
            "memory": "memory_ram",
            "temperature": "operating_temperature",
            "operating_temp": "operating_temperature",
            "can": "can_interface",
            "can_bus": "can_interface",
            "power": "power_consumption",
            "ota": "ota_supported",
            "ai": "npu",
            "command": "npu_enable_command",
        }
        value = aliases.get(clean, clean)
        if value in SPEC_FIELDS and value not in normalized:
            normalized.append(value)
    return normalized or list(SPEC_FIELDS)


def spec_to_dict(spec: ECUSpec) -> dict[str, Any]:
    return {
        "model": spec.model,
        "series": spec.series,
        "source": spec.source,
        "processor": spec.processor,
        "memory_ram": spec.memory_ram,
        "storage": spec.storage,
        "can_interface": spec.can_interface,
        "ethernet": spec.ethernet,
        "power_consumption": spec.power_consumption,
        "operating_temperature": spec.operating_temperature,
        "connectors": spec.connectors,
        "npu": spec.npu,
        "ota_supported": spec.ota_supported,
        "npu_enable_command": spec.npu_enable_command,
        "safety": spec.safety,
    }


def retrieval_results_from_tool(result: ToolResult) -> list[RetrievalResult]:
    if result.name != "search_documents" or not isinstance(result.result, list):
        return []
    retrievals: list[RetrievalResult] = []
    for item in result.result:
        retrievals.append(
            RetrievalResult(
                content=str(item.get("content", "")),
                metadata=dict(item.get("metadata") or {}),
                score=float(item.get("score") or 0.0),
            )
        )
    return retrievals


def _unique_sources(values) -> list[str]:
    return sorted({str(value) for value in values if value})
