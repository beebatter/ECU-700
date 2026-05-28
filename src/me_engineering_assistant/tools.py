from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Sequence

from me_engineering_assistant.coverage import model_field_evidence
from me_engineering_assistant.documents import CatalogEntry, ModelFieldEvidence
from me_engineering_assistant.retriever import InMemoryECURetriever, RetrievalResult


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
    """Small RAG toolbox shared by the local agent and MCP server."""

    def __init__(
        self,
        retriever: InMemoryECURetriever,
        *,
        catalog: Sequence[CatalogEntry] | None = None,
        field_table: Sequence[ModelFieldEvidence] | None = None,
    ) -> None:
        self.retriever = retriever
        self.catalog = list(catalog or ())
        self.field_table = list(field_table or ())

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_documents",
                "description": "Search ECU markdown documentation chunks for grounding evidence.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "models": {"type": "array", "items": {"type": "string"}},
                        "series": {"type": "array", "items": {"type": "string"}},
                        "field": {"type": "string"},
                        "fields": {"type": "array", "items": {"type": "string"}},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "list_sources",
                "description": "List source documents currently available to the RAG retriever.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_document_catalog",
                "description": "Return the indexed ECU document catalog.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_model_field_evidence",
                "description": "Return structured model-field evidence extracted from source documents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "models": {"type": "array", "items": {"type": "string"}},
                        "field": {"type": "string"},
                    },
                },
            },
            {
                "name": "check_evidence_coverage",
                "description": "Check whether model-field evidence exists for each requested model.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "models": {"type": "array", "items": {"type": "string"}},
                        "field": {"type": "string"},
                    },
                    "required": ["models", "field"],
                },
            },
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name == "search_documents":
            return self.search_documents(**call.arguments)
        if call.name == "list_sources":
            return self.list_sources()
        if call.name == "get_document_catalog":
            return self.get_document_catalog()
        if call.name == "get_model_field_evidence":
            return self.get_model_field_evidence(**call.arguments)
        if call.name == "check_evidence_coverage":
            return self.check_evidence_coverage(**call.arguments)
        return ToolResult(
            name=call.name,
            arguments=call.arguments,
            result={"error": f"Unknown tool: {call.name}"},
            sources=[],
        )

    def search_documents(
        self,
        query: str,
        *,
        models: Sequence[str] | None = None,
        series: Sequence[str] | None = None,
        field: str | None = None,
        fields: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        top_k: int = 6,
    ) -> ToolResult:
        filters = {}
        if models:
            filters["models"] = list(models)
        if series:
            filters["series"] = list(series)
        selected_fields = list(fields or ())
        if field:
            selected_fields.append(field)
        if selected_fields:
            filters["fields"] = _unique_sources(selected_fields)
        if sources:
            filters["sources"] = list(sources)

        if models and len(models) > 1:
            results = self._retrieve_diverse_by_model(query=query, models=models, filters=filters, top_k=top_k)
        else:
            results = self.retriever.retrieve(query, filters=filters or None, top_k=top_k)
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
            arguments={
                "query": query,
                "models": list(models or ()),
                "series": list(series or ()),
                "fields": selected_fields,
                "sources": list(sources or ()),
                "top_k": top_k,
            },
            result=payload,
            sources=_unique_sources(result.metadata.get("source") for result in results),
        )

    def get_document_catalog(self) -> ToolResult:
        rows = [entry.to_dict() for entry in self.catalog]
        return ToolResult(
            name="get_document_catalog",
            arguments={},
            result=rows,
            sources=[row["source"] for row in rows],
        )

    def get_model_field_evidence(
        self,
        models: Sequence[str] | None = None,
        field: str | None = None,
    ) -> ToolResult:
        rows = model_field_evidence(self.field_table, models=models, field=field)
        return ToolResult(
            name="get_model_field_evidence",
            arguments={"models": list(models or ()), "field": field or ""},
            result=rows,
            sources=_unique_sources(row.get("source") for row in rows),
        )

    def check_evidence_coverage(self, models: Sequence[str], field: str) -> ToolResult:
        rows = model_field_evidence(self.field_table, models=models, field=field)
        covered = {row["model"] for row in rows}
        missing = [{"model": model, "field": field} for model in models if model not in covered]
        result = {
            "complete": not missing,
            "field": field,
            "models": list(models),
            "covered_models": sorted(covered),
            "missing": missing,
            "evidence": rows,
        }
        return ToolResult(
            name="check_evidence_coverage",
            arguments={"models": list(models), "field": field},
            result=result,
            sources=_unique_sources(row.get("source") for row in rows),
        )

    def list_sources(self) -> ToolResult:
        rows_by_source: dict[str, dict[str, str]] = {}
        for chunk in self.retriever.chunks:
            source = chunk.metadata.get("source", "unknown")
            rows_by_source.setdefault(
                source,
                {
                    "source": source,
                    "model": chunk.metadata.get("model", "unknown"),
                    "series": chunk.metadata.get("series", "unknown"),
                    "title": chunk.metadata.get("title", source),
                },
            )
        rows = sorted(rows_by_source.values(), key=lambda row: row["source"])
        return ToolResult(
            name="list_sources",
            arguments={},
            result=rows,
            sources=[row["source"] for row in rows],
        )

    def _retrieve_diverse_by_model(
        self,
        *,
        query: str,
        models: Sequence[str],
        filters: dict[str, Sequence[str]],
        top_k: int,
    ) -> list[RetrievalResult]:
        per_model = max(2, ceil(top_k / len(models)))
        candidates: list[RetrievalResult] = []
        for model in models:
            model_filters = {**filters, "models": [model]}
            candidates.extend(self.retriever.retrieve(query, filters=model_filters, top_k=per_model))

        seen: set[tuple[str, str]] = set()
        deduped: list[RetrievalResult] = []
        for result in sorted(candidates, key=lambda item: item.score, reverse=True):
            key = (result.metadata.get("source", ""), result.content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
            if len(deduped) >= top_k:
                break
        return deduped


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

