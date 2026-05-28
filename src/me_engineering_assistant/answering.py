from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from me_engineering_assistant.config import bool_env, env
from me_engineering_assistant.coverage import CoverageReport, check_plan_coverage
from me_engineering_assistant.llm import chat_with_configured_llm, json_with_configured_llm
from me_engineering_assistant.planner import QueryPlan, models_in_query
from me_engineering_assistant.retriever import RetrievalResult
from me_engineering_assistant.tools import ECUToolbox, ToolCall, ToolResult, retrieval_results_from_tool
from me_engineering_assistant.visualization import trace_step


@dataclass(frozen=True)
class AnswerDraft:
    answer: str
    confidence: float
    sources: list[str]
    plan: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] | None = None
    trace: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class AgentPlan:
    rationale: str
    calls: list[ToolCall]


def generate_answer(
    query: str,
    toolbox: ECUToolbox,
    retrieved: Sequence[RetrievalResult],
    route: Mapping[str, Any] | None = None,
    query_plan: QueryPlan | None = None,
) -> AnswerDraft:
    plan = plan_tool_calls(query=query, toolbox=toolbox, route=route, query_plan=query_plan)
    trace = [
        trace_step(
            "plan",
            summary=plan.rationale,
            query_plan=query_plan.to_dict() if query_plan else None,
            tool_calls=[tool_call_to_dict(call) for call in plan.calls],
        )
    ]
    tool_results = execute_plan(toolbox, plan)
    if not any(result.name == "search_documents" for result in tool_results):
        tool_results.append(toolbox.execute(_search_call(query, route, query_plan=query_plan)))

    trace.append(
        trace_step(
            "execute_tools",
            summary=f"Executed {len(tool_results)} tool call(s).",
            tools=[tool_result_summary(result) for result in tool_results],
        )
    )

    evidence = evidence_from_tool_results(tool_results) or list(retrieved)
    coverage_report = check_plan_coverage(query_plan, evidence) if query_plan else None
    corrective_results = []
    if coverage_report and not coverage_report.complete:
        corrective_results = corrective_retrieval(toolbox=toolbox, query_plan=query_plan, missing=coverage_report.missing)
        tool_results.extend(corrective_results)
        evidence = evidence_from_tool_results(tool_results) or list(retrieved)
        coverage_report = check_plan_coverage(query_plan, evidence)
    if coverage_report:
        trace.append(
            trace_step(
                "coverage_check",
                summary=(
                    "Coverage complete."
                    if coverage_report.complete
                    else f"Coverage incomplete for {len(coverage_report.missing)} entity-field pair(s)."
                ),
                coverage=coverage_report.to_dict(),
                corrective_calls=len(corrective_results),
            )
        )

    evidence = coverage_approved_evidence(evidence, coverage_report)
    evidence_sources = _sources_from_evidence(evidence)
    answer = None
    synthesis_mode = "extractive_rag"
    if bool_env("ME_USE_LLM_ANSWER", default=bool(env("DEEPSEEK_API_KEY"))) or bool_env("ME_FORCE_LLM", default=False):
        answer = compose_with_llm(query=query, plan=plan, evidence=evidence, coverage_report=coverage_report)
        synthesis_mode = "llm_grounded" if answer else "extractive_after_llm_fallback"
    if not answer:
        answer = compose_from_evidence(query=query, evidence=evidence, query_plan=query_plan, coverage_report=coverage_report)
    trace.append(
        trace_step(
            "synthesize",
            summary=f"Generated draft answer with {synthesis_mode}.",
            mode=synthesis_mode,
            sources=evidence_sources,
        )
    )

    verified_answer, confidence = apply_grounding_checks(
        query=query,
        answer=answer,
        sources=evidence_sources,
        tool_results=tool_results,
        evidence=evidence,
        coverage_report=coverage_report,
    )
    trace.append(
        trace_step(
            "grounding",
            summary="Checked generated answer against retrieved evidence.",
            confidence=confidence,
            sources=evidence_sources,
        )
    )
    return AnswerDraft(
        answer=verified_answer,
        confidence=confidence,
        sources=evidence_sources,
        plan=plan_to_dict(plan),
        tool_results=[result.to_dict() for result in tool_results],
        trace=trace,
    )


def plan_tool_calls(
    *,
    query: str,
    toolbox: ECUToolbox,
    route: Mapping[str, Any] | None = None,
    query_plan: QueryPlan | None = None,
) -> AgentPlan:
    use_llm_planner = bool_env("ME_USE_LLM_PLANNER", default=bool(env("DEEPSEEK_API_KEY")))
    use_llm_tool_planner = use_llm_planner and bool_env("ME_USE_LLM_TOOL_PLANNER", default=True)
    if use_llm_tool_planner:
        llm_plan = plan_with_llm(query=query, toolbox=toolbox, route=route)
        if llm_plan is not None:
            sanitized = sanitize_plan(query=query, route=route, toolbox=toolbox, plan=llm_plan)
            if sanitized.calls:
                return ensure_search_call(query=query, route=route, query_plan=query_plan, plan=sanitized)

    calls: list[ToolCall] = []
    if query_plan and query_plan.entities and query_plan.attribute:
        calls.extend(
            [
                ToolCall("get_model_field_evidence", {"models": query_plan.entities, "field": query_plan.attribute}),
                ToolCall("check_evidence_coverage", {"models": query_plan.entities, "field": query_plan.attribute}),
            ]
        )
    calls.append(_search_call(query, route, query_plan=query_plan))
    plan = AgentPlan(rationale="RAG controller selected evidence retrieval tools.", calls=calls)
    return sanitize_plan(query=query, route=route, toolbox=toolbox, plan=plan)


def plan_with_llm(
    *,
    query: str,
    toolbox: ECUToolbox,
    route: Mapping[str, Any] | None = None,
) -> AgentPlan | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a ReAct/Plan-Execute controller for an ECU engineering RAG agent. "
                "Choose retrieval tool calls that gather evidence from internal markdown documents. "
                "Do not answer directly. Return strict JSON only with keys rationale and tool_calls. "
                "Each tool call must have name and arguments. Available tools:\n"
                f"{json.dumps(toolbox.manifest(), indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"query": query, "router_hint": route or {}}, ensure_ascii=False),
        },
    ]
    payload = json_with_configured_llm(messages)
    if not isinstance(payload, dict):
        return None
    calls = []
    for item in payload.get("tool_calls", []):
        if not isinstance(item, dict):
            continue
        calls.append(ToolCall(name=str(item.get("name")), arguments=dict(item.get("arguments") or {})))
    return AgentPlan(rationale=str(payload.get("rationale") or "LLM planned retrieval calls."), calls=calls)


def complete_plan(query: str, route: Mapping[str, Any] | None, plan: AgentPlan) -> AgentPlan:
    calls = list(plan.calls)
    if not any(call.name == "search_documents" for call in calls):
        calls.insert(0, _search_call(query, route))
    return AgentPlan(rationale=f"{plan.rationale} Plan normalized to the RAG retrieval contract.", calls=calls)


def ensure_search_call(
    *,
    query: str,
    route: Mapping[str, Any] | None,
    query_plan: QueryPlan | None,
    plan: AgentPlan,
) -> AgentPlan:
    calls = list(plan.calls)
    if not any(call.name == "search_documents" for call in calls):
        calls.append(_search_call(query, route, query_plan=query_plan))
    return AgentPlan(rationale=f"{plan.rationale} Plan normalized to the RAG retrieval contract.", calls=calls)


def sanitize_plan(
    *,
    query: str,
    route: Mapping[str, Any] | None,
    toolbox: ECUToolbox,
    plan: AgentPlan,
) -> AgentPlan:
    manifest = {tool["name"]: tool for tool in toolbox.manifest()}
    sanitized: list[ToolCall] = []
    for call in plan.calls:
        if call.name not in manifest:
            continue
        allowed = set((manifest[call.name].get("input_schema") or {}).get("properties") or {})
        arguments = {key: value for key, value in call.arguments.items() if key in allowed}
        if call.name == "search_documents":
            arguments = _sanitize_search_arguments(query=query, route=route, arguments=arguments)
        elif call.name == "list_sources":
            arguments = {}
        sanitized.append(ToolCall(name=call.name, arguments=arguments))
    return AgentPlan(rationale=plan.rationale, calls=sanitized)


def _sanitize_search_arguments(
    *,
    query: str,
    route: Mapping[str, Any] | None,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {"query": str(arguments.get("query") or query)}
    route_models = list((route or {}).get("models") or ())
    for key in ("models", "series", "fields", "sources"):
        values = _as_string_list(arguments.get(key))
        if values:
            sanitized[key] = values
    if arguments.get("field"):
        sanitized["field"] = str(arguments["field"])
    if route_models and not sanitized.get("models"):
        sanitized["models"] = route_models
    sanitized["top_k"] = _clamp_top_k(arguments.get("top_k"), default=10)
    return sanitized


def execute_plan(toolbox: ECUToolbox, plan: AgentPlan) -> list[ToolResult]:
    results: list[ToolResult] = []
    allowed = {tool["name"] for tool in toolbox.manifest()}
    for call in plan.calls:
        if call.name not in allowed:
            continue
        results.append(toolbox.execute(call))
    return results


def plan_to_dict(plan: AgentPlan) -> dict[str, Any]:
    return {
        "rationale": plan.rationale,
        "tool_calls": [tool_call_to_dict(call) for call in plan.calls],
    }


def tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": call.arguments}


def tool_result_summary(result: ToolResult) -> dict[str, Any]:
    payload = result.result
    count = len(payload) if isinstance(payload, list) else 1
    return {
        "name": result.name,
        "arguments": result.arguments,
        "sources": result.sources,
        "result_count": count,
    }


def compose_with_llm(
    query: str,
    plan: AgentPlan,
    evidence: Sequence[RetrievalResult],
    coverage_report: CoverageReport | None = None,
) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an ECU engineering assistant using standard RAG. Answer only from the "
                "provided retrieved markdown evidence. Do not use outside knowledge. If the evidence "
                "does not support the user's criterion, say the documentation does not contain enough "
                "information. Include concise source filenames in the answer."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": query,
                    "plan_rationale": plan.rationale,
                    "coverage": coverage_report.to_dict() if coverage_report else None,
                    "retrieved_evidence": [evidence_to_dict(item) for item in evidence],
                },
                ensure_ascii=False,
            ),
        },
    ]
    return chat_with_configured_llm(messages, temperature=0.0)


def compose_from_evidence(
    query: str,
    evidence: Sequence[RetrievalResult],
    query_plan: QueryPlan | None = None,
    coverage_report: CoverageReport | None = None,
) -> str:
    if coverage_report and coverage_report.items:
        return compose_from_coverage(query_plan=query_plan, coverage_report=coverage_report)

    structured_answer = compose_structured_evidence_answer(evidence=evidence, query_plan=query_plan)
    if structured_answer:
        return structured_answer

    missing_models = requested_models_without_evidence(query, evidence)
    if missing_models:
        return (
            "The internal ECU documentation does not contain enough evidence for "
            f"{', '.join(missing_models)}."
        )
    if not evidence:
        return "I could not find enough evidence in the internal ECU documentation to answer this question."

    bullets = []
    for item in evidence[:10]:
        source = item.metadata.get("source", "unknown")
        excerpt = _evidence_excerpt(item.content)
        bullets.append(f"- Source: {source}\n  Evidence: {excerpt}")
    return "Relevant documentation evidence:\n" + "\n".join(bullets)


def compose_structured_evidence_answer(
    *,
    evidence: Sequence[RetrievalResult],
    query_plan: QueryPlan | None = None,
) -> str | None:
    rows = [item for item in evidence if item.metadata.get("chunk_type") == "field" and item.metadata.get("field")]
    if not rows:
        return None

    entities = query_plan.entities if query_plan else []
    if entities:
        rows = [item for item in rows if item.metadata.get("model") in entities]
    if not rows:
        return None

    by_field: dict[str, list[RetrievalResult]] = {}
    for item in rows:
        field = item.metadata.get("field", "")
        by_field.setdefault(field, [])
        by_field[field].append(item)

    selected = select_field_groups(by_field, limit=8, entities=entities)
    lines = ["Grounded answer from retrieved ECU documentation:"]
    for field, field_rows in selected:
        rows_by_model: dict[str, RetrievalResult] = {}
        for row in sorted(field_rows, key=lambda item: item.score, reverse=True):
            rows_by_model.setdefault(row.metadata.get("model", ""), row)
        if not rows_by_model:
            continue

        ordered_models = entities or sorted(rows_by_model)
        values = []
        for entity in ordered_models:
            row = rows_by_model.get(entity)
            if row is None:
                continue
            value = row.metadata.get("value") or _evidence_excerpt(row.content, max_chars=120)
            source = row.metadata.get("source", "unknown")
            values.append(f"{entity}: {value} (Source: {source})")
        if values:
            label = field_rows[0].metadata.get("field_label") or field.replace("_", " ")
            lines.append(f"- {label}: " + "; ".join(values))
    for context in nearby_section_context(evidence):
        source = context.metadata.get("source", "unknown")
        lines.append(f"- Context from {source}: {_evidence_excerpt(context.content, max_chars=420)}")
    return "\n".join(lines) if len(lines) > 1 else None


def nearby_section_context(evidence: Sequence[RetrievalResult], *, max_items: int = 2) -> list[RetrievalResult]:
    if not evidence:
        return []
    top_score = max(item.score for item in evidence)
    contexts = [
        item
        for item in evidence
        if item.metadata.get("chunk_type") == "section" and item.score >= top_score - 0.08
    ]
    return sorted(contexts, key=lambda item: item.score, reverse=True)[:max_items]


def select_field_groups(
    by_field: Mapping[str, Sequence[RetrievalResult]],
    *,
    limit: int,
    entities: Sequence[str] | None = None,
) -> list[tuple[str, Sequence[RetrievalResult]]]:
    ranked = sorted(by_field.items(), key=_field_group_score, reverse=True)
    if not ranked:
        return []

    top_score = _field_group_score(ranked[0])[0]
    if _field_group_score(ranked[0])[1] > 1:
        return ranked[:1]
    if entities:
        if len(entities) > 1:
            return ranked[:limit]
        close_groups = [item for item in ranked if top_score - _field_group_score(item)[0] <= 0.12]
        return (close_groups or ranked)[:limit]
    close_groups = [item for item in ranked if top_score - _field_group_score(item)[0] <= 0.12]
    return (close_groups or ranked)[:limit]


def compose_from_coverage(query_plan: QueryPlan | None, coverage_report: CoverageReport) -> str:
    attribute = coverage_report.attribute or (query_plan.attribute if query_plan else "requested field")
    if not coverage_report.complete:
        missing = ", ".join(f"{item['entity']} {item['attribute']}" for item in coverage_report.missing)
        return f"The internal ECU documentation does not contain enough evidence for: {missing}."

    title = f"{attribute.replace('_', ' ').title()} evidence"
    lines = [f"{title}:"]
    for item in coverage_report.items:
        if not item.evidence:
            continue
        evidence = item.evidence[0]
        value = evidence.get("value") or _evidence_excerpt(evidence.get("content", ""), max_chars=220)
        source = evidence.get("source", "unknown")
        lines.append(f"- {item.entity}: {value} (Source: {source})")
    return "\n".join(lines)


def apply_grounding_checks(
    *,
    query: str,
    answer: str,
    sources: list[str],
    tool_results: Sequence[ToolResult],
    evidence: Sequence[RetrievalResult],
    coverage_report: CoverageReport | None = None,
) -> tuple[str, float]:
    has_missing_evidence = "not contain enough" in answer.lower() or "could not find" in answer.lower()
    confidence = 0.45 if has_missing_evidence else 0.62
    if sources and evidence and not has_missing_evidence:
        confidence = 0.9
    if _contains_numeric_or_command(answer) and sources and not has_missing_evidence:
        confidence = 0.95

    unsupported_claims = unsupported_numeric_or_command_claims(answer, tool_results, evidence)
    if unsupported_claims:
        confidence = min(confidence, 0.5)
        answer = (
            f"{answer}\nGrounding warning: I could not verify "
            f"{', '.join(unsupported_claims)} in the retrieved evidence."
        )

    if requested_models_without_evidence(query, evidence):
        confidence = min(confidence, 0.45)
    if coverage_report and coverage_report.items:
        confidence = min(confidence, 0.45) if not coverage_report.complete else max(confidence, 0.95)
    if sources and "source:" not in answer.lower() and "source" not in answer.lower():
        answer = f"{answer}\nsource: {', '.join(sources)}"
    return answer, confidence


def evidence_from_tool_results(tool_results: Sequence[ToolResult]) -> list[RetrievalResult]:
    evidence: list[RetrievalResult] = []
    for result in tool_results:
        evidence.extend(retrieval_results_from_tool(result))
        evidence.extend(structured_retrieval_results_from_tool(result))
    return _dedupe_evidence(evidence)


def structured_retrieval_results_from_tool(result: ToolResult) -> list[RetrievalResult]:
    if result.name != "get_model_field_evidence" or not isinstance(result.result, list):
        return []
    retrievals = []
    for row in result.result:
        if not isinstance(row, dict):
            continue
        metadata = {str(key): str(value) for key, value in row.items()}
        metadata["chunk_type"] = "field"
        content = (
            f"Model: {metadata.get('model', '')}\n"
            f"Field: {metadata.get('field_label') or metadata.get('field', '')}\n"
            f"Value: {metadata.get('value', '')}\n"
            f"Source: {metadata.get('source', '')}"
        )
        retrievals.append(RetrievalResult(content=content, metadata=metadata, score=1.0))
    return retrievals


def evidence_to_dict(item: RetrievalResult) -> dict[str, Any]:
    return {
        "content": item.content,
        "metadata": item.metadata,
        "score": item.score,
    }


def requested_models_without_evidence(query: str, evidence: Sequence[RetrievalResult]) -> list[str]:
    requested = models_in_query(query)
    if not requested:
        return []
    available = {item.metadata.get("model") for item in evidence}
    return [model for model in requested if model not in available]


def _search_call(query: str, route: Mapping[str, Any] | None, query_plan: QueryPlan | None = None) -> ToolCall:
    models = list((route or {}).get("models") or ())
    arguments: dict[str, Any] = {"query": query, "models": models, "top_k": 10}
    if query_plan and query_plan.attribute:
        arguments["field"] = query_plan.attribute
    return ToolCall("search_documents", arguments)


def corrective_retrieval(
    *,
    toolbox: ECUToolbox,
    query_plan: QueryPlan,
    missing: Sequence[Mapping[str, str]],
) -> list[ToolResult]:
    results = []
    for item in missing:
        entity = item.get("entity") or item.get("model")
        attribute = item.get("attribute") or query_plan.attribute
        if not entity or not attribute:
            continue
        results.append(
            toolbox.search_documents(
                query=f"{entity} {attribute.replace('_', ' ')} specification",
                models=[entity],
                field=attribute,
                top_k=3,
            )
        )
    return results


def coverage_approved_evidence(
    evidence: Sequence[RetrievalResult],
    coverage_report: CoverageReport | None,
) -> list[RetrievalResult]:
    if not coverage_report or not coverage_report.items:
        return list(evidence)
    approved = []
    for item in coverage_report.items:
        for result in evidence:
            if result.metadata.get("model") == item.entity and result.metadata.get("field"):
                approved.append(result)
    return _dedupe_evidence(approved)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item]
    return [str(value)]


def _clamp_top_k(value: Any, default: int) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        top_k = default
    return max(1, min(10, top_k))


def _dedupe_evidence(evidence: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[tuple[str, str]] = set()
    deduped = []
    for item in evidence:
        key = (item.metadata.get("source", ""), item.content)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _sources_from_evidence(evidence: Sequence[RetrievalResult]) -> list[str]:
    return sorted({item.metadata.get("source", "unknown") for item in evidence})


def _evidence_excerpt(content: str, max_chars: int = 1_400) -> str:
    excerpt = re.sub(r"\s+", " ", content).strip()
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 3].rstrip() + "..."


def _field_group_score(item: tuple[str, Sequence[RetrievalResult]]) -> tuple[float, int]:
    _field, rows = item
    best_score = max((row.score for row in rows), default=0.0)
    model_count = len({row.metadata.get("model", "") for row in rows})
    return best_score, model_count


def _contains_numeric_or_command(answer: str) -> bool:
    return bool(re.search(r"\d|me-driver-ctl", answer))


def unsupported_numeric_or_command_claims(
    answer: str,
    tool_results: Sequence[ToolResult],
    evidence: Sequence[RetrievalResult],
) -> list[str]:
    evidence_text = json.dumps([result.to_dict() for result in tool_results], ensure_ascii=False).lower()
    evidence_text += "\n" + "\n".join(item.content for item in evidence).lower()
    claims = _dedupe(re.findall(r"me-driver-ctl\s+[a-z0-9_\-= ]+", answer.lower()))
    claims.extend(
        _dedupe(
            re.findall(
                r"\+?-?\d+(?:\.\d+)?\s*(?:gb|kb|mb|mbps|tops|ghz|mhz|ma|a|v|°c|c)\b",
                answer.lower(),
            )
        )
    )
    unsupported = []
    normalized_evidence = evidence_text.replace("°", "")
    for claim in claims:
        compact = claim.strip().rstrip(".,;")
        if compact and compact.replace("°", "") not in normalized_evidence:
            unsupported.append(compact)
    return _dedupe(unsupported)


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
