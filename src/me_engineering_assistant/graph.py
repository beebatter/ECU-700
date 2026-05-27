from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from me_engineering_assistant.answering import AnswerDraft, generate_answer, models_in_query
from me_engineering_assistant.documents import chunk_documents, load_source_documents
from me_engineering_assistant.knowledge import ECUSpec, extract_specs
from me_engineering_assistant.review import maybe_enqueue_review
from me_engineering_assistant.retriever import InMemoryECURetriever, RetrievalResult
from me_engineering_assistant.tools import ECUToolbox
from me_engineering_assistant.visualization import trace_step


ALL_MODELS = ("ECU-750", "ECU-850", "ECU-850b")


class GraphState(TypedDict, total=False):
    query: str
    route: dict[str, Any]
    retrieved: list[dict[str, Any]]
    draft: dict[str, Any]
    answer: str
    sources: list[str]
    confidence: float
    needs_review: bool
    review_id: str | None
    review_reason: str | None
    trace: list[dict[str, Any]]


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    models: list[str]
    series: list[str]
    reasons: list[str]

    def filters(self) -> dict[str, Sequence[str]]:
        filters: dict[str, Sequence[str]] = {}
        if self.models:
            filters["models"] = self.models
        if self.series and not self.models:
            filters["series"] = self.series
        return filters


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    sources: list[str]
    confidence: float
    route: RouteDecision
    needs_review: bool = False
    review_id: str | None = None
    review_reason: str | None = None
    trace: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "route": asdict(self.route),
            "needs_review": self.needs_review,
            "review_id": self.review_id,
            "review_reason": self.review_reason,
        }
        if self.trace is not None:
            payload["trace"] = self.trace
        return payload


def route_query(query: str) -> RouteDecision:
    lowered = query.lower()
    models = models_in_query(query)
    reasons: list[str] = []

    if models:
        reasons.append("explicit_model_reference")

    if _is_global_query(lowered):
        models = list(ALL_MODELS)
        reasons.append("global_or_availability_question")
    elif _is_comparison_query(lowered):
        if not models:
            models = list(ALL_MODELS)
        reasons.append("comparison_question")
    elif "800 series" in lowered:
        models = ["ECU-850", "ECU-850b"]
        reasons.append("series_reference")

    series = sorted({model_to_series(model) for model in models})
    mode = "single"
    if len(models) > 1:
        mode = "multi_source"
    if set(models) == set(ALL_MODELS):
        mode = "all_models"

    return RouteDecision(mode=mode, models=models, series=series, reasons=reasons or ["semantic_retrieval"])


def model_to_series(model: str) -> str:
    return "ECU-700" if model == "ECU-750" else "ECU-800"


class ECUAgent:
    def __init__(
        self,
        docs_dir: str | Path | None = None,
        prefer_langchain: bool = True,
        embedding_endpoint: str | None = None,
    ) -> None:
        self.docs_dir = Path(docs_dir).expanduser().resolve() if docs_dir else None
        self.documents = load_source_documents(base_path=self.docs_dir)
        self.chunks = chunk_documents(self.documents)
        self.specs: dict[str, ECUSpec] = extract_specs(self.documents)
        self.retriever = InMemoryECURetriever(
            self.chunks,
            prefer_langchain=prefer_langchain,
            embedding_endpoint=embedding_endpoint,
        )
        self.toolbox = ECUToolbox(self.specs, self.retriever)
        self._workflow = self._build_langgraph_workflow()

    def answer(self, query: str, include_trace: bool = False) -> AgentResponse:
        if self._workflow is not None:
            state = self._workflow.invoke({"query": query})
        else:
            state = self._run_without_langgraph({"query": query})

        route = RouteDecision(**state["route"])
        return AgentResponse(
            answer=state["answer"],
            sources=list(state["sources"]),
            confidence=float(state["confidence"]),
            route=route,
            needs_review=bool(state.get("needs_review", False)),
            review_id=state.get("review_id"),
            review_reason=state.get("review_reason"),
            trace=list(state.get("trace", [])) if include_trace else None,
        )

    def invoke(self, query: str) -> dict[str, Any]:
        return self.answer(query).to_dict()

    def _run_without_langgraph(self, state: GraphState) -> GraphState:
        for node in (self._route_node, self._retrieve_node, self._generate_node, self._validate_node):
            state = node(state)
        return state

    def _build_langgraph_workflow(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(GraphState)
        graph.add_node("route_query", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("validate", self._validate_node)
        graph.set_entry_point("route_query")
        graph.add_edge("route_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", END)
        return graph.compile()

    def _route_node(self, state: GraphState) -> GraphState:
        route = route_query(state["query"])
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "route_query",
                summary=f"Route mode {route.mode}; models: {', '.join(route.models) or 'semantic retrieval'}.",
                route=asdict(route),
            )
        )
        return {**state, "route": asdict(route), "trace": trace}

    def _retrieve_node(self, state: GraphState) -> GraphState:
        route = RouteDecision(**state["route"])
        results = self.retriever.retrieve(state["query"], filters=route.filters(), top_k=6)
        retrieved = [result.to_dict() for result in results]
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "retrieve",
                summary=f"Retrieved {len(results)} evidence chunk(s).",
                backend=self.retriever.backend_name,
                sources=sorted({result.metadata.get("source", "unknown") for result in results}),
                top_scores=[round(result.score, 4) for result in results[:3]],
            )
        )
        return {**state, "retrieved": retrieved, "trace": trace}

    def _generate_node(self, state: GraphState) -> GraphState:
        retrieved = [_result_from_dict(item) for item in state["retrieved"]]
        draft = generate_answer(
            state["query"],
            self.toolbox,
            retrieved,
            route=state.get("route"),
        )
        trace = list(state.get("trace", []))
        trace.extend(draft.trace or [])
        return {**state, "draft": asdict(draft), "trace": trace}

    def _validate_node(self, state: GraphState) -> GraphState:
        draft = AnswerDraft(**state["draft"])
        retrieved = [_result_from_dict(item) for item in state.get("retrieved", [])]
        sources = draft.sources or sorted(
            {item["metadata"].get("source", "unknown") for item in state.get("retrieved", [])}
        )
        confidence = max(0.0, min(1.0, draft.confidence))
        review = maybe_enqueue_review(
            query=state["query"],
            answer=draft.answer,
            confidence=confidence,
            sources=sources,
            route=state.get("route", {}),
            retrieved=retrieved,
        )
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "validate",
                summary="Validated confidence, sources, and review routing.",
                confidence=confidence,
                needs_review=review.needs_review,
                review_id=review.review_id,
                review_reason=review.reason,
            )
        )
        return {
            **state,
            "answer": draft.answer,
            "sources": sources,
            "confidence": confidence,
            "needs_review": review.needs_review,
            "review_id": review.review_id,
            "review_reason": review.reason,
            "trace": trace,
        }


def _result_from_dict(item: Mapping[str, Any]) -> RetrievalResult:
    return RetrievalResult(
        content=str(item["content"]),
        metadata=dict(item["metadata"]),
        score=float(item["score"]),
    )


def _is_comparison_query(query: str) -> bool:
    return bool(re.search(r"\b(compare|comparison|differences?|versus|vs)\b", query))


def _is_global_query(query: str) -> bool:
    global_terms = ("across all", "all ecu", "which ecu", "which models", "support ota", "supports ota")
    return any(term in query for term in global_terms)
