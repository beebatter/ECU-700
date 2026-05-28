from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Sequence

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

    def __init__(self, retriever: InMemoryECURetriever) -> None:
        self.retriever = retriever

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
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name == "search_documents":
            return self.search_documents(**call.arguments)
        if call.name == "list_sources":
            return self.list_sources()
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
        sources: Sequence[str] | None = None,
        top_k: int = 6,
    ) -> ToolResult:
        filters = {}
        if models:
            filters["models"] = list(models)
        if sources:
            filters["sources"] = list(sources)

        if models and len(models) > 1:
            results = self._retrieve_diverse_by_model(query=query, models=models, top_k=top_k)
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
                "sources": list(sources or ()),
                "top_k": top_k,
            },
            result=payload,
            sources=_unique_sources(result.metadata.get("source") for result in results),
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
        top_k: int,
    ) -> list[RetrievalResult]:
        per_model = max(2, ceil(top_k / len(models)))
        candidates: list[RetrievalResult] = []
        for model in models:
            candidates.extend(self.retriever.retrieve(query, filters={"models": [model]}, top_k=per_model))

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
