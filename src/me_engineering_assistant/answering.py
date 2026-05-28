from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from me_engineering_assistant.config import bool_env
from me_engineering_assistant.llm import chat_with_configured_llm, json_with_configured_llm
from me_engineering_assistant.retriever import RetrievalResult
from me_engineering_assistant.tools import ECUToolbox, ToolCall, ToolResult, normalize_fields, normalize_model_name
from me_engineering_assistant.visualization import trace_step


MODEL_ORDER = ("ECU-750", "ECU-850", "ECU-850b")


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


@dataclass(frozen=True)
class QueryIntent:
    """General user intent, independent of any golden test question."""

    operation: str
    models: list[str]
    fields: list[str]
    rank_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "models": self.models,
            "fields": self.fields,
            "rank_direction": self.rank_direction,
        }


def generate_answer(
    query: str,
    toolbox: ECUToolbox,
    retrieved: Sequence[RetrievalResult],
    route: Mapping[str, Any] | None = None,
) -> AnswerDraft:
    intent = infer_query_intent(query, route)
    plan = plan_tool_calls(query=query, toolbox=toolbox, route=route, intent=intent)
    trace = [
        trace_step(
            "plan",
            summary=plan.rationale,
            intent=intent.to_dict(),
            tool_calls=[tool_call_to_dict(call) for call in plan.calls],
        )
    ]
    tool_results = execute_plan(toolbox, plan)
    if not any(result.name == "search_documents" for result in tool_results):
        tool_results.append(
            toolbox.execute(
                ToolCall(
                    "search_documents",
                    {"query": query, "models": intent.models, "top_k": 4},
                )
            )
        )

    trace.append(
        trace_step(
            "execute_tools",
            summary=f"Executed {len(tool_results)} tool call(s).",
            tools=[tool_result_summary(result) for result in tool_results],
        )
    )

    evidence_sources = _sources_from_results(tool_results) or _sources_from_retrieval(retrieved)
    answer = None
    synthesis_mode = "generic_evidence_composer"
    if bool_env("ME_USE_LLM_ANSWER", default=False) or bool_env("ME_FORCE_LLM", default=False):
        answer = compose_with_llm(query=query, plan=plan, tool_results=tool_results, intent=intent)
        synthesis_mode = "llm_grounded" if answer else "generic_after_llm_fallback"
    if not answer:
        answer = compose_from_evidence(query=query, tool_results=tool_results, intent=intent)
    trace.append(
        trace_step(
            "synthesize",
            summary=f"Generated draft answer with {synthesis_mode}.",
            mode=synthesis_mode,
            intent=intent.to_dict(),
        )
    )

    verified_answer, confidence = apply_grounding_checks(
        answer=answer,
        sources=evidence_sources,
        tool_results=tool_results,
    )
    trace.append(
        trace_step(
            "grounding",
            summary="Checked generated answer against retrieved tool evidence.",
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
    intent: QueryIntent | None = None,
) -> AgentPlan:
    intent = intent or infer_query_intent(query, route)
    if bool_env("ME_USE_LLM_PLANNER", default=False):
        llm_plan = plan_with_llm(query=query, toolbox=toolbox, route=route, intent=intent)
        if llm_plan and llm_plan.calls:
            return complete_plan(query=query, route=route, plan=llm_plan, intent=intent)
    return fallback_plan(query=query, route=route, intent=intent)


def plan_with_llm(
    *,
    query: str,
    toolbox: ECUToolbox,
    route: Mapping[str, Any] | None = None,
    intent: QueryIntent | None = None,
) -> AgentPlan | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a ReAct/Plan-Execute controller for an ECU engineering RAG agent. "
                "Infer the user's intent and choose tool calls that can answer the question "
                "from internal ECU documentation. Do not answer directly. Return strict JSON "
                "only with keys rationale and tool_calls. Each tool call must have name and "
                "arguments. Available tools:\n"
                f"{json.dumps(toolbox.manifest(), indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "router_hint": route or {},
                    "local_intent_hint": (intent or infer_query_intent(query, route)).to_dict(),
                },
                ensure_ascii=False,
            ),
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
    return AgentPlan(rationale=str(payload.get("rationale") or "LLM planned tool calls."), calls=calls)


def fallback_plan(
    query: str,
    route: Mapping[str, Any] | None = None,
    intent: QueryIntent | None = None,
) -> AgentPlan:
    intent = intent or infer_query_intent(query, route)
    calls = [ToolCall("search_documents", {"query": query, "models": intent.models, "top_k": 4})]
    if intent.operation in {"compare", "rank", "filter"} and intent.models:
        calls.append(ToolCall("compare_model_specs", {"models": intent.models, "fields": intent.fields}))
    elif len(intent.models) == 1:
        calls.append(ToolCall("read_model_spec", {"model": intent.models[0]}))
    elif intent.models:
        calls.append(ToolCall("compare_model_specs", {"models": intent.models, "fields": intent.fields}))
    else:
        calls.append(ToolCall("list_models", {}))
    return AgentPlan(rationale="Local intent planner selected evidence tools.", calls=calls)


def complete_plan(
    query: str,
    route: Mapping[str, Any] | None,
    plan: AgentPlan,
    intent: QueryIntent | None = None,
) -> AgentPlan:
    intent = intent or infer_query_intent(query, route)
    calls = list(plan.calls)
    tool_names = [call.name for call in calls]
    if "search_documents" not in tool_names:
        calls.insert(0, ToolCall("search_documents", {"query": query, "models": intent.models, "top_k": 4}))
    needs_structured_specs = intent.operation in {"compare", "rank", "filter"} or len(intent.models) > 1
    if needs_structured_specs and intent.models and "compare_model_specs" not in tool_names:
        calls.append(ToolCall("compare_model_specs", {"models": intent.models, "fields": intent.fields}))
    elif len(intent.models) == 1 and "read_model_spec" not in tool_names:
        calls.append(ToolCall("read_model_spec", {"model": intent.models[0]}))
    return AgentPlan(rationale=f"{plan.rationale} Plan validated against inferred intent.", calls=calls)


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
    tool_results: Sequence[ToolResult],
    intent: QueryIntent,
) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an ECU engineering assistant. Answer only using the provided tool evidence. "
                "Every factual claim must be traceable to a source filename in the evidence. "
                "If evidence is missing, say the documentation does not contain enough information. "
                "Do not invent specifications."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": query,
                    "inferred_intent": intent.to_dict(),
                    "plan_rationale": plan.rationale,
                    "tool_results": [result.to_dict() for result in tool_results],
                },
                ensure_ascii=False,
            ),
        },
    ]
    return chat_with_configured_llm(messages, temperature=0.0)


def compose_from_evidence(
    *,
    query: str,
    tool_results: Sequence[ToolResult],
    intent: QueryIntent | None = None,
) -> str:
    unknown_models = unknown_models_in_query(query)
    if unknown_models:
        available = _models_from_tool_results(tool_results) or list(MODEL_ORDER)
        return (
            f"The internal documentation covers {', '.join(available)}, but it does not contain "
            f"enough information for {', '.join(unknown_models)}. I cannot answer that from the "
            "current ECU documents."
        )

    intent = intent or infer_query_intent(query)
    rows = _evidence_rows(tool_results)
    if rows:
        fields = _select_fields(intent.fields, rows)
        if intent.operation == "rank":
            return _compose_rank_response(rows, fields, intent.rank_direction)
        if intent.operation == "filter":
            return _compose_filter_response(rows, fields)
        if intent.operation == "compare" or len(rows) > 1:
            return _compose_comparison_response(rows, fields)
        return _compose_lookup_response(rows[0], fields)

    search = _first_result(tool_results, "search_documents")
    if search and isinstance(search.result, list) and search.result:
        excerpts = " ".join(_one_line(str(item.get("content", ""))) for item in search.result[:2])
        return f"Based on retrieved ECU documentation: {excerpts}"
    return "I could not find enough evidence in the internal ECU documentation to answer this question."


def _compose_lookup_response(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    model = str(row.get("model") or "The requested ECU")
    facts = _field_facts(row, fields)
    source = row.get("source")
    if not facts:
        return f"{model} evidence was found, but not for the requested field. Source: {source}."
    if len(facts) == 1:
        return f"{model} {facts[0]}. Source: {source}."
    return f"{model}: " + "; ".join(facts) + f". Source: {source}."


def _compose_comparison_response(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    parts = []
    for row in rows:
        facts = _field_facts(row, fields)
        if facts:
            parts.append(f"{row.get('model')}: " + ", ".join(facts))
    if not parts:
        return "The internal documentation did not contain enough comparable evidence for the requested fields."
    return "Comparison: " + "; ".join(parts) + f". Sources: {_format_sources(rows)}."


def _compose_filter_response(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    bool_field = next(
        (
            field
            for field in fields
            if any(isinstance(row.get(field), bool) for row in rows)
        ),
        None,
    )
    if not bool_field:
        return _compose_comparison_response(rows, fields)

    subject = _boolean_subject(bool_field)
    supported = [str(row.get("model")) for row in rows if row.get(bool_field) is True]
    unsupported = [str(row.get("model")) for row in rows if row.get(bool_field) is False]
    parts = []
    if supported:
        parts.append(f"{subject} are supported by {', '.join(supported)}")
    if unsupported:
        parts.append(f"{', '.join(unsupported)} does not support {subject}")
    if not parts:
        return f"The internal documentation did not contain enough {subject} evidence."
    return "; ".join(parts) + f". Sources: {_format_sources(rows)}."


def _compose_rank_response(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    direction: str | None,
) -> str:
    field = fields[0] if fields else "operating_temperature"
    scored = []
    for row in rows:
        value = row.get(field)
        score = _score_value(value, direction or "max")
        if score is not None:
            scored.append((score, row, value))
    if not scored:
        return f"The internal documentation does not contain enough {human_field_phrase(field)} evidence."

    winner_score = min(score for score, _, _ in scored) if direction == "min" else max(score for score, _, _ in scored)
    winners = [row for score, row, _ in scored if score == winner_score]
    winner_models = ", ".join(str(row.get("model")) for row in winners)
    values = "; ".join(
        f"{row.get('model')}: {value}"
        for _, row, value in scored
    )
    label = "Lowest" if direction == "min" else "Highest"
    return (
        f"{label} {human_field_phrase(field)}: {winner_models}. "
        f"Retrieved values: {values}. Sources: {_format_sources(rows)}."
    )


def apply_grounding_checks(
    *,
    answer: str,
    sources: list[str],
    tool_results: Sequence[ToolResult],
) -> tuple[str, float]:
    evidence_count = sum(1 for result in tool_results if result.sources)
    has_missing_evidence = "not contain enough" in answer.lower() or "could not find" in answer.lower()
    confidence = 0.62
    if sources and evidence_count:
        confidence = 0.9
    if _contains_numeric_or_command(answer) and sources:
        confidence = 0.95
    if has_missing_evidence:
        confidence = 0.45
    unsupported_claims = unsupported_numeric_or_command_claims(answer, tool_results)
    if unsupported_claims:
        confidence = min(confidence, 0.5)
        answer = (
            f"{answer} Grounding warning: I could not verify "
            f"{', '.join(unsupported_claims)} in the retrieved evidence."
        )
    if sources and "source" not in answer.lower():
        answer = f"{answer} Sources: {', '.join(sources)}."
    return answer, confidence


def infer_query_intent(
    query: str,
    route: Mapping[str, Any] | None = None,
) -> QueryIntent:
    models = list((route or {}).get("models") or models_in_query(query))
    if not models and _mentions_all_models(query):
        models = list(MODEL_ORDER)

    fields = infer_fields_from_query(query)
    operation = "lookup"
    rank_direction = _rank_direction(query)
    if rank_direction and len(models) != 1:
        operation = "rank"
    elif _is_filter_query(query):
        operation = "filter"
    elif _is_comparison_query(query) or len(models) > 1:
        operation = "compare"
    elif not models and _is_inventory_query(query):
        operation = "enumerate"

    return QueryIntent(
        operation=operation,
        models=models,
        fields=fields,
        rank_direction=rank_direction,
    )


def infer_fields_from_query(query: str) -> list[str]:
    lowered = query.lower()
    fields = []
    keyword_map = (
        (("enable", "npu"), "npu_enable_command"),
        (("ai", "capab"), "npu"),
        (("npu",), "npu"),
        (("ram",), "memory_ram"),
        (("memory",), "memory_ram"),
        (("storage",), "storage"),
        (("can_bus",), "can_interface"),
        (("power",), "power_consumption"),
        (("load",), "power_consumption"),
        (("temperature",), "operating_temperature"),
        (("temp",), "operating_temperature"),
        (("harshest",), "operating_temperature"),
        (("ota",), "ota_supported"),
        (("over-the-air",), "ota_supported"),
        (("processor",), "processor"),
        (("clock",), "processor"),
    )
    for keywords, field in keyword_map:
        if all(_keyword_present(lowered, keyword) for keyword in keywords) and field not in fields:
            fields.append(field)
    if _is_difference_query(query) and not fields:
        fields = ["processor", "npu", "memory_ram", "storage", "power_consumption"]
    return fields or ["processor", "memory_ram", "storage", "can_interface", "operating_temperature"]


def models_in_query(query: str) -> list[str]:
    lowered = query.lower()
    models: list[str] = []
    if re.search(r"\becu[-\s]?750\b", lowered) or re.search(r"\becu[-\s]?700\b", lowered):
        models.append("ECU-750")
    if re.search(r"\becu[-\s]?(850b|800b)\b", lowered) or "npu" in lowered:
        models.append("ECU-850b")
    if re.search(r"\becu[-\s]?(850|800a|800)\b(?!b)", lowered):
        models.append("ECU-850")
    return _dedupe(models)


def unknown_models_in_query(query: str) -> list[str]:
    known = set(models_in_query(query))
    models = []
    for match in re.findall(r"\becu[-\s]?(\d+[a-z]?)\b", query.lower()):
        if match in {"700", "800", "800a", "800b"}:
            continue
        normalized = normalize_model_name(f"ECU-{match}")
        if normalized not in known and normalized not in MODEL_ORDER:
            models.append(normalized)
    return _dedupe(models)


def human_field_phrase(field: str) -> str:
    return {
        "processor": "processor",
        "memory_ram": "RAM",
        "storage": "storage",
        "can_interface": "CAN bus capability",
        "ethernet": "Ethernet",
        "power_consumption": "power consumption",
        "operating_temperature": "operating temperature",
        "connectors": "connectors",
        "npu": "AI/NPU capability",
        "ota_supported": "OTA support",
        "npu_enable_command": "NPU enable command",
        "safety": "safety certification",
    }.get(field, field.replace("_", " "))


def _evidence_rows(tool_results: Sequence[ToolResult]) -> list[dict[str, Any]]:
    compare = _first_result(tool_results, "compare_model_specs")
    if compare and isinstance(compare.result, list):
        return _dedupe_rows(compare.result)

    rows: list[Mapping[str, Any]] = []
    for result in tool_results:
        payload = result.result
        if result.name == "read_model_spec" and isinstance(payload, dict) and payload.get("model"):
            rows.append(payload)
        elif result.name == "list_models" and isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))
    return _dedupe_rows(rows)


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        model = str(row.get("model") or "")
        if not model or model in seen or row.get("error"):
            continue
        seen.add(model)
        deduped.append(dict(row))
    return deduped


def _select_fields(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    selected = normalize_fields(fields)
    available = [field for field in selected if any(row.get(field) is not None for row in rows)]
    return available or selected


def _field_facts(row: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    facts = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        facts.append(f"{human_field_phrase(field)}: {_format_value(field, value)}")
    return facts


def _format_value(field: str, value: Any) -> str:
    if isinstance(value, bool):
        return "supported" if value else "does not support"
    if field == "npu_enable_command":
        return f"`{value}`"
    return str(value)


def _boolean_subject(field: str) -> str:
    return {"ota_supported": "OTA updates"}.get(field, human_field_phrase(field))


def _format_sources(rows: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(sorted({str(row.get("source")) for row in rows if row.get("source")}))


def _first_result(tool_results: Sequence[ToolResult], name: str) -> ToolResult | None:
    for result in tool_results:
        if result.name == name:
            return result
    return None


def _sources_from_results(tool_results: Sequence[ToolResult]) -> list[str]:
    return sorted({source for result in tool_results for source in result.sources})


def _sources_from_retrieval(retrieved: Sequence[RetrievalResult]) -> list[str]:
    return sorted({result.metadata.get("source", "unknown") for result in retrieved})


def _models_from_tool_results(tool_results: Sequence[ToolResult]) -> list[str]:
    models: list[str] = []
    for result in tool_results:
        payload = result.result
        if isinstance(payload, dict) and payload.get("model"):
            models.append(str(payload["model"]))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("model"):
                    models.append(str(item["model"]))
    return _dedupe(models)


def _mentions_all_models(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("all ecu", "all models", "across all", "which ecu", "which model"))


def _is_comparison_query(query: str) -> bool:
    return bool(re.search(r"\b(compare|comparison|differences?|versus|vs)\b", query.lower()))


def _is_difference_query(query: str) -> bool:
    return bool(re.search(r"\b(differences?|different|upgrade|compare|versus|vs)\b", query.lower()))


def _is_filter_query(query: str) -> bool:
    lowered = query.lower()
    return bool(re.search(r"\b(support|supports|supported|available|capable)\b", lowered))


def _is_inventory_query(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("what models", "list models", "available models", "what ecu"))


def _keyword_present(lowered_query: str, keyword: str) -> bool:
    if keyword == "can_bus":
        return bool(re.search(r"\bcan\s+(bus|fd|interface|capab)", lowered_query))
    if keyword in {"capab", "over-the-air"}:
        return keyword in lowered_query
    return bool(re.search(rf"\b{re.escape(keyword)}\b", lowered_query))


def _rank_direction(query: str) -> str | None:
    lowered = query.lower()
    min_terms = ("lowest", "minimum", "smallest", "least", "coldest")
    max_terms = ("highest", "maximum", "largest", "most", "harshest", "best")
    if any(term in lowered for term in min_terms):
        return "min"
    if any(term in lowered for term in max_terms):
        return "max"
    if re.search(r"\bwhich\s+(ecu|model).*\b(can|has|operates?)\b", lowered):
        return "max"
    return None


def _score_value(value: Any, direction: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    scores = [_normalize_number(number, unit) for number, unit in _number_unit_pairs(str(value))]
    if not scores:
        return None
    return min(scores) if direction == "min" else max(scores)


def _number_unit_pairs(value: str) -> list[tuple[float, str]]:
    pairs = []
    for number, unit in re.findall(
        r"([+-]?\d+(?:\.\d+)?)\s*(gb|mb|mbps|tops|ghz|mhz|ma|a|v|°c|c)?",
        value.lower(),
    ):
        pairs.append((float(number), unit))
    return pairs


def _normalize_number(number: float, unit: str) -> float:
    multipliers = {
        "gb": 1024.0,
        "mb": 1.0,
        "ghz": 1000.0,
        "mhz": 1.0,
        "a": 1000.0,
        "ma": 1.0,
    }
    return number * multipliers.get(unit, 1.0)


def _contains_numeric_or_command(answer: str) -> bool:
    return bool(re.search(r"\d|me-driver-ctl", answer))


def unsupported_numeric_or_command_claims(answer: str, tool_results: Sequence[ToolResult]) -> list[str]:
    evidence = json.dumps([result.to_dict() for result in tool_results], ensure_ascii=False).lower()
    claims = _dedupe(re.findall(r"me-driver-ctl\s+[a-z0-9_\-= ]+", answer.lower()))
    claims.extend(
        _dedupe(
            re.findall(
                r"\+?-?\d+(?:\.\d+)?\s*(?:gb|mb|mbps|tops|ghz|ma|a|v|°c|c)\b",
                answer.lower(),
            )
        )
    )
    unsupported = []
    normalized_evidence = evidence.replace("°", "")
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


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
