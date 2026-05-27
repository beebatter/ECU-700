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


def generate_answer(
    query: str,
    toolbox: ECUToolbox,
    retrieved: Sequence[RetrievalResult],
    route: Mapping[str, Any] | None = None,
) -> AnswerDraft:
    plan = plan_tool_calls(query=query, toolbox=toolbox, route=route)
    trace = [
        trace_step(
            "plan",
            summary=plan.rationale,
            tool_calls=[tool_call_to_dict(call) for call in plan.calls],
        )
    ]
    tool_results = execute_plan(toolbox, plan)
    if not any(result.name == "search_documents" for result in tool_results):
        fallback_search = ToolCall(
            "search_documents",
            {"query": query, "models": list((route or {}).get("models") or ()), "top_k": 4},
        )
        tool_results.append(toolbox.execute(fallback_search))

    trace.append(
        trace_step(
            "execute_tools",
            summary=f"Executed {len(tool_results)} tool call(s).",
            tools=[tool_result_summary(result) for result in tool_results],
        )
    )

    evidence_sources = _sources_from_results(tool_results) or _sources_from_retrieval(retrieved)
    answer = None
    synthesis_mode = "deterministic"
    if bool_env("ME_USE_LLM_ANSWER", default=False) or bool_env("ME_FORCE_LLM", default=False):
        answer = synthesize_with_llm(query=query, plan=plan, tool_results=tool_results)
        synthesis_mode = "llm_grounded" if answer else "deterministic_after_llm_fallback"
    if not answer:
        answer = synthesize_from_tool_results(query=query, tool_results=tool_results)
    trace.append(
        trace_step(
            "synthesize",
            summary=f"Generated draft answer with {synthesis_mode} synthesis.",
            mode=synthesis_mode,
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
) -> AgentPlan:
    if bool_env("ME_USE_LLM_PLANNER", default=False):
        llm_plan = plan_with_llm(query=query, toolbox=toolbox, route=route)
        if llm_plan and llm_plan.calls:
            return complete_plan(query=query, route=route, plan=llm_plan)
    return fallback_plan(query=query, route=route)


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
                "Choose tool calls that can answer the question from internal ECU documentation. "
                "Do not answer directly. Return strict JSON only with keys rationale and tool_calls. "
                "Each tool call must have name and arguments. Available tools:\n"
                f"{json.dumps(toolbox.manifest(), indent=2)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"query": query, "router_hint": route or {}},
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


def fallback_plan(query: str, route: Mapping[str, Any] | None = None) -> AgentPlan:
    models = list((route or {}).get("models") or models_in_query(query))
    if not models and _mentions_all_models(query):
        models = list(MODEL_ORDER)
    fields = infer_fields_from_query(query)
    calls = [ToolCall("search_documents", {"query": query, "models": models, "top_k": 4})]
    if len(models) == 1 and not _is_comparison_query(query):
        calls.append(ToolCall("read_model_spec", {"model": models[0]}))
    elif models:
        calls.append(ToolCall("compare_model_specs", {"models": models, "fields": fields}))
    else:
        calls.append(ToolCall("list_models", {}))
    return AgentPlan(rationale="Local fallback planner selected evidence tools.", calls=calls)


def complete_plan(query: str, route: Mapping[str, Any] | None, plan: AgentPlan) -> AgentPlan:
    models = list((route or {}).get("models") or models_in_query(query))
    if not models and _mentions_all_models(query):
        models = list(MODEL_ORDER)

    calls = list(plan.calls)
    tool_names = [call.name for call in calls]
    fields = infer_fields_from_query(query)
    if "search_documents" not in tool_names:
        calls.insert(0, ToolCall("search_documents", {"query": query, "models": models, "top_k": 4}))
    if len(models) > 1 and "compare_model_specs" not in tool_names:
        calls.append(ToolCall("compare_model_specs", {"models": models, "fields": fields}))
    elif len(models) == 1 and not _is_comparison_query(query) and "read_model_spec" not in tool_names:
        calls.append(ToolCall("read_model_spec", {"model": models[0]}))
    return AgentPlan(rationale=f"{plan.rationale} Plan validated against router intent.", calls=calls)


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


def synthesize_with_llm(query: str, plan: AgentPlan, tool_results: Sequence[ToolResult]) -> str | None:
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
                    "plan_rationale": plan.rationale,
                    "tool_results": [result.to_dict() for result in tool_results],
                },
                ensure_ascii=False,
            ),
        },
    ]
    return chat_with_configured_llm(messages, temperature=0.0)


def synthesize_from_tool_results(
    *,
    query: str,
    tool_results: Sequence[ToolResult],
) -> str:
    unknown_models = unknown_models_in_query(query)
    if unknown_models:
        available = _models_from_tool_results(tool_results) or list(MODEL_ORDER)
        return (
            f"The internal documentation covers {', '.join(available)}, but it does not contain "
            f"enough information for {', '.join(unknown_models)}. I cannot answer that from the "
            "current ECU documents."
        )

    comparison = _first_result(tool_results, "compare_model_specs")
    single_spec = _first_result(tool_results, "read_model_spec")
    if comparison and isinstance(comparison.result, list):
        return synthesize_comparison(query=query, rows=comparison.result, fields=comparison.arguments.get("fields", []))
    if single_spec and isinstance(single_spec.result, dict):
        return synthesize_single_spec(query=query, spec=single_spec.result)
    search = _first_result(tool_results, "search_documents")
    if search and isinstance(search.result, list) and search.result:
        excerpts = " ".join(_one_line(str(item.get("content", ""))) for item in search.result[:2])
        return f"Based on retrieved ECU documentation: {excerpts}"
    return "I could not find enough evidence in the internal ECU documentation to answer this question."


def synthesize_single_spec(query: str, spec: Mapping[str, Any]) -> str:
    fields = infer_fields_from_query(query)
    source = spec.get("source")
    model = spec.get("model", "The requested ECU")
    if "npu_enable_command" in fields and spec.get("npu_enable_command"):
        return f"To enable the NPU on the {model}, run `{spec['npu_enable_command']}`. Source: {source}."
    if "npu" in fields and spec.get("npu"):
        return (
            f"The {model} includes a dedicated Neural Processing Unit (NPU), {spec['npu']}, "
            f"for AI acceleration. Source: {source}."
        )
    if "ota_supported" in fields:
        support = "supports" if spec.get("ota_supported") else "does not support"
        return f"The {model} {support} OTA updates. Source: {source}."
    values = [(field, spec.get(field)) for field in fields if spec.get(field) is not None]
    if values:
        field, value = values[0]
        return f"The {model} {human_field_phrase(field)} is {value}. Source: {source}."
    return f"{model} specification evidence was found, but not for the requested field. Source: {source}."


def synthesize_comparison(query: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    selected_fields = normalize_fields(fields or infer_fields_from_query(query))
    if _asks_harshest(query) and "operating_temperature" in selected_fields:
        return synthesize_harshest_temperature(rows)
    if selected_fields == ["ota_supported"]:
        supported = [row["model"] for row in rows if row.get("ota_supported") is True]
        unsupported = [row["model"] for row in rows if row.get("ota_supported") is False]
        parts = []
        if supported:
            parts.append(f"OTA updates are supported by {', '.join(supported)}")
        if unsupported:
            parts.append(f"{', '.join(unsupported)} does not support OTA updates")
        return "; ".join(parts) + f". Sources: {format_sources_from_rows(rows)}."

    if _is_difference_query(query) and len(rows) == 2:
        diffs = []
        for field in selected_fields:
            first = rows[0].get(field)
            second = rows[1].get(field)
            if first != second and (first is not None or second is not None):
                diffs.append(f"{human_field_phrase(field)} ({rows[0]['model']}: {first}; {rows[1]['model']}: {second})")
        if diffs:
            return "Key differences are: " + "; ".join(diffs) + f". Sources: {format_sources_from_rows(rows)}."

    row_phrases = []
    for row in rows:
        values = [
            f"{human_field_phrase(field)}: {row.get(field)}"
            for field in selected_fields
            if row.get(field) is not None
        ]
        if values:
            row_phrases.append(f"{row['model']}: " + ", ".join(values))
    if row_phrases:
        return "; ".join(row_phrases) + f". Sources: {format_sources_from_rows(rows)}."
    return "The internal documentation did not contain enough comparable evidence for the requested fields."


def synthesize_harshest_temperature(rows: Sequence[Mapping[str, Any]]) -> str:
    scored = []
    for row in rows:
        high = _temperature_high(str(row.get("operating_temperature") or ""))
        if high is not None:
            scored.append((high, row))
    if not scored:
        return "The internal documentation does not contain enough temperature evidence to compare models."
    best_high = max(score for score, _ in scored)
    best = [row for score, row in scored if score == best_high]
    best_models = " and ".join(row["model"] for row in best)
    baseline = "; ".join(
        f"{row['model']}: {row.get('operating_temperature')}" for _, row in scored
    )
    return (
        f"{best_models} can operate in the harshest high-temperature conditions. "
        f"Temperature ranges: {baseline}. Sources: {format_sources_from_rows(rows)}."
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
        (("can",), "can_interface"),
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
        if all(keyword in lowered for keyword in keywords) and field not in fields:
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


def format_sources_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
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
    return any(term in lowered for term in ("all ecu", "all models", "across all", "which ecu"))


def _is_comparison_query(query: str) -> bool:
    return bool(re.search(r"\b(compare|comparison|differences?|versus|vs)\b", query.lower()))


def _is_difference_query(query: str) -> bool:
    return bool(re.search(r"\b(differences?|different|upgrade|compare|versus|vs)\b", query.lower()))


def _asks_harshest(query: str) -> bool:
    return "harshest" in query.lower() or "which ecu can operate" in query.lower()


def _temperature_high(value: str) -> int | None:
    match = re.search(r"to\s*\+?(-?\d+)", value)
    return int(match.group(1)) if match else None


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
