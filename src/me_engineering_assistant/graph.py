from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from me_engineering_assistant.answering import AnswerDraft, generate_answer
from me_engineering_assistant.config import bool_env
from me_engineering_assistant.conversation import DEFAULT_SESSION_ID, ConversationManager
from me_engineering_assistant.documents import (
    default_document_paths,
    build_document_catalog,
    build_model_field_table,
    chunk_documents,
    load_source_documents,
)
from me_engineering_assistant.llm import get_llm_call_stats, reset_llm_call_stats
from me_engineering_assistant.memory import GLOBAL_SCOPE, MemoryStore
from me_engineering_assistant.observability import log_agent_response
from me_engineering_assistant.planner import QueryPlan, models_from_catalog, plan_query
from me_engineering_assistant.review import maybe_enqueue_review
from me_engineering_assistant.retriever import InMemoryECURetriever, RetrievalResult
from me_engineering_assistant.tools import ECUToolbox
from me_engineering_assistant.visualization import trace_step


class GraphState(TypedDict, total=False):
    query: str
    effective_query: str
    session_id: str
    memory_context: dict[str, Any]
    query_plan: dict[str, Any]
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
    metrics: dict[str, float]


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
    llm_calls: int = 0
    llm_latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    generation_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "route": asdict(self.route),
            "needs_review": self.needs_review,
            "review_id": self.review_id,
            "review_reason": self.review_reason,
            "llm_calls": self.llm_calls,
            "llm_latency_seconds": self.llm_latency_seconds,
            "retrieval_latency_seconds": self.retrieval_latency_seconds,
            "generation_latency_seconds": self.generation_latency_seconds,
            "total_latency_seconds": self.total_latency_seconds,
        }
        if self.trace is not None:
            payload["trace"] = self.trace
        return payload


@dataclass(frozen=True)
class CorpusArtifacts:
    documents: list[Any]
    chunks: list[Any]
    catalog: list[Any]
    field_table: list[Any]


def route_query(query: str) -> RouteDecision:
    corpus = _load_corpus(None)
    catalog = corpus.catalog
    plan = plan_query(query, llm_enabled=False, catalog=catalog)
    return _route_from_plan(plan, catalog=catalog)


def model_to_series(model: str, catalog=None) -> str:
    for entry in catalog or ():
        if entry.model == model:
            return entry.series
    return "unknown"


class ECUAgent:
    def __init__(
        self,
        docs_dir: str | Path | None = None,
        *,
        memory_enabled: bool | None = None,
        memory_store: MemoryStore | None = None,
        memory_path: str | Path | None = None,
        memory_scope: str = GLOBAL_SCOPE,
    ) -> None:
        self.docs_dir = Path(docs_dir).expanduser().resolve() if docs_dir else None
        corpus = _load_corpus(self.docs_dir)
        self.documents = corpus.documents
        self.chunks = corpus.chunks
        self.catalog = corpus.catalog
        self.field_table = corpus.field_table
        self.retriever = InMemoryECURetriever(self.chunks)
        self.toolbox = ECUToolbox(self.retriever, catalog=self.catalog, field_table=self.field_table)
        self.memory_enabled = bool_env("ME_MEMORY_ENABLED", default=False) if memory_enabled is None else memory_enabled
        self.conversation: ConversationManager | None = None
        if self.memory_enabled:
            self.conversation = ConversationManager(
                memory_store or MemoryStore(memory_path),
                scope=memory_scope,
            )
        self._workflow = self._build_langgraph_workflow()

    def answer(
        self,
        query: str,
        include_trace: bool = False,
        session_id: str | None = None,
    ) -> AgentResponse:
        reset_llm_call_stats()
        started = time.perf_counter()
        initial_state: GraphState = {"query": query, "session_id": session_id or DEFAULT_SESSION_ID}
        if self._workflow is not None:
            state = self._workflow.invoke(initial_state)
        else:
            state = self._run_without_langgraph(initial_state)

        route = RouteDecision(**state["route"])
        trace = list(state.get("trace", []))
        llm_stats = get_llm_call_stats()
        metrics = state.get("metrics", {})
        total_latency = time.perf_counter() - started
        response = AgentResponse(
            answer=state["answer"],
            sources=list(state["sources"]),
            confidence=float(state["confidence"]),
            route=route,
            needs_review=bool(state.get("needs_review", False)),
            review_id=state.get("review_id"),
            review_reason=state.get("review_reason"),
            trace=trace if include_trace else None,
            llm_calls=llm_stats.count,
            llm_latency_seconds=llm_stats.latency_seconds,
            retrieval_latency_seconds=float(metrics.get("retrieval_latency_seconds", 0.0)),
            generation_latency_seconds=float(metrics.get("generation_latency_seconds", 0.0)),
            total_latency_seconds=total_latency,
        )
        log_agent_response(
            query=query,
            response=response.to_dict(),
            trace=trace,
            latency_seconds=total_latency,
            retriever_backend=self.retriever.backend_name,
            docs_dir=str(self.docs_dir) if self.docs_dir else None,
        )
        return response

    def invoke(self, query: str) -> dict[str, Any]:
        return self.answer(query).to_dict()

    def _run_without_langgraph(self, state: GraphState) -> GraphState:
        for node in (
            self._memory_node,
            self._route_node,
            self._retrieve_node,
            self._generate_node,
            self._validate_node,
            self._reflect_memory_node,
        ):
            state = node(state)
        return state

    def _build_langgraph_workflow(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(GraphState)
        graph.add_node("load_memory", self._memory_node)
        graph.add_node("route_query", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("reflect_memory", self._reflect_memory_node)
        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "route_query")
        graph.add_edge("route_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", "reflect_memory")
        graph.add_edge("reflect_memory", END)
        return graph.compile()

    def _memory_node(self, state: GraphState) -> GraphState:
        if self.conversation is None:
            return state

        session_id = str(state.get("session_id") or DEFAULT_SESSION_ID)
        context = self.conversation.build_context(state["query"], session_id=session_id)
        effective_query = self.conversation.enrich_query(state["query"], context)
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "load_memory",
                summary=(
                    f"Loaded memory for session {session_id}: "
                    f"{len(context.recent_turns)} recent turn(s), {len(context.memories)} long-term memory item(s)."
                ),
                session_id=session_id,
                has_summary=bool(context.summary),
                recent_turns=len(context.recent_turns),
                memories=[{"kind": memory.kind, "score": round(memory.score, 4)} for memory in context.memories],
            )
        )
        return {
            **state,
            "session_id": session_id,
            "memory_context": context.to_dict(),
            "effective_query": effective_query,
            "trace": trace,
        }

    def _route_node(self, state: GraphState) -> GraphState:
        query_plan = plan_query(
            str(state.get("effective_query") or state["query"]),
            catalog=self.catalog,
            field_table=self.field_table,
            tool_manifest=self.toolbox.manifest(),
        )
        route = _route_from_plan(query_plan, catalog=self.catalog)
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "route_query",
                summary=f"Route mode {route.mode}; models: {', '.join(route.models) or 'semantic retrieval'}.",
                route=asdict(route),
                query_plan=query_plan.to_dict(),
            )
        )
        return {**state, "query_plan": query_plan.to_dict(), "route": asdict(route), "trace": trace}

    def _retrieve_node(self, state: GraphState) -> GraphState:
        route = RouteDecision(**state["route"])
        query = str(state.get("effective_query") or state["query"])
        started = time.perf_counter()
        results = self.retriever.retrieve(query, filters=route.filters(), top_k=6)
        latency = time.perf_counter() - started
        retrieved = [result.to_dict() for result in results]
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "retrieve",
                summary=f"Retrieved {len(results)} evidence chunk(s).",
                backend=self.retriever.backend_name,
                sources=sorted({result.metadata.get("source", "unknown") for result in results}),
                top_scores=[round(result.score, 4) for result in results[:3]],
                latency_seconds=latency,
            )
        )
        return {
            **state,
            "retrieved": retrieved,
            "trace": trace,
            "metrics": _merge_metrics(state, retrieval_latency_seconds=latency),
        }

    def _generate_node(self, state: GraphState) -> GraphState:
        retrieved = [_result_from_dict(item) for item in state["retrieved"]]
        started = time.perf_counter()
        draft = generate_answer(
            state["query"],
            self.toolbox,
            retrieved,
            route=state.get("route"),
            query_plan=QueryPlan(**state["query_plan"]) if state.get("query_plan") else None,
        )
        latency = time.perf_counter() - started
        trace = list(state.get("trace", []))
        trace.extend(draft.trace or [])
        return {
            **state,
            "draft": asdict(draft),
            "trace": trace,
            "metrics": _merge_metrics(state, generation_latency_seconds=latency),
        }

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
                metrics={
                    **state.get("metrics", {}),
                    "llm_calls": get_llm_call_stats().count,
                    "llm_latency_seconds": get_llm_call_stats().latency_seconds,
                },
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

    def _reflect_memory_node(self, state: GraphState) -> GraphState:
        if self.conversation is None:
            return state

        memory_context = state.get("memory_context") or {}
        result = self.conversation.record_turn(
            session_id=str(state.get("session_id") or DEFAULT_SESSION_ID),
            query=state["query"],
            answer=state["answer"],
            confidence=float(state["confidence"]),
            sources=list(state["sources"]),
            previous_summary=str(memory_context.get("summary") or ""),
        )
        trace = list(state.get("trace", []))
        trace.append(
            trace_step(
                "reflect_memory",
                summary=(
                    "Recorded turn and updated session memory; "
                    f"stored {len(result.stored_memory_ids)} long-term memory item(s)."
                ),
                turn_index=result.turn.turn_index,
                stored_memory_ids=result.stored_memory_ids,
                updated_summary=bool(result.reflection.session_summary),
            )
        )
        return {**state, "trace": trace}


def _result_from_dict(item: Mapping[str, Any]) -> RetrievalResult:
    return RetrievalResult(
        content=str(item["content"]),
        metadata=dict(item["metadata"]),
        score=float(item["score"]),
    )


def _route_from_plan(plan: QueryPlan, catalog=None) -> RouteDecision:
    series = sorted({value for model in plan.entities if (value := model_to_series(model, catalog=catalog)) != "unknown"})
    known_models = models_from_catalog(catalog)
    if not plan.entities:
        mode = "semantic"
    elif set(plan.entities) == set(known_models):
        mode = "all_indexed_models"
    else:
        mode = "metadata_filtered"
    return RouteDecision(mode=mode, models=plan.entities, series=series, reasons=plan.reasons)


_CORPUS_CACHE: dict[tuple[tuple[str, int, int], ...], CorpusArtifacts] = {}


def _load_corpus(docs_dir: str | Path | None) -> CorpusArtifacts:
    key = _corpus_cache_key(docs_dir)
    cached = _CORPUS_CACHE.get(key)
    if cached is not None:
        return cached

    documents = load_source_documents(base_path=docs_dir)
    chunks = chunk_documents(documents)
    corpus = CorpusArtifacts(
        documents=documents,
        chunks=chunks,
        catalog=build_document_catalog(documents),
        field_table=build_model_field_table(chunks),
    )
    _CORPUS_CACHE[key] = corpus
    return corpus


def _corpus_cache_key(docs_dir: str | Path | None) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for path in default_document_paths(docs_dir):
        resolved = path.expanduser().resolve()
        try:
            stat = resolved.stat()
        except OSError:
            rows.append((str(resolved), 0, 0))
            continue
        rows.append((str(resolved), stat.st_mtime_ns, stat.st_size))
    return tuple(rows)


def _merge_metrics(state: Mapping[str, Any], **values: float) -> dict[str, float]:
    metrics = {str(key): float(value) for key, value in (state.get("metrics") or {}).items()}
    metrics.update({key: float(value) for key, value in values.items()})
    return metrics
