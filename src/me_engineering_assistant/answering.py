from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from me_engineering_assistant.config import bool_env
from me_engineering_assistant.llm import chat_with_configured_llm, json_with_configured_llm
from me_engineering_assistant.retriever import RetrievalResult
from me_engineering_assistant.tools import ECUToolbox, ToolCall, ToolResult, retrieval_results_from_tool
from me_engineering_assistant.visualization import trace_step


MODEL_ALIASES = {
    "ECU-700": "ECU-750",
    "ECU-800": "ECU-850",
    "ECU-800A": "ECU-850",
    "ECU-800B": "ECU-850b",
}


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
        tool_results.append(toolbox.execute(_search_call(query, route)))

    trace.append(
        trace_step(
            "execute_tools",
            summary=f"Executed {len(tool_results)} tool call(s).",
            tools=[tool_result_summary(result) for result in tool_results],
        )
    )

    evidence = evidence_from_tool_results(tool_results) or list(retrieved)
    evidence_sources = _sources_from_evidence(evidence)
    answer = None
    synthesis_mode = "extractive_rag"
    if bool_env("ME_USE_LLM_ANSWER", default=False) or bool_env("ME_FORCE_LLM", default=False):
        answer = compose_with_llm(query=query, plan=plan, evidence=evidence)
        synthesis_mode = "llm_grounded" if answer else "extractive_after_llm_fallback"
    if not answer:
        answer = compose_from_evidence(query=query, evidence=evidence)
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
) -> AgentPlan:
    if bool_env("ME_USE_LLM_PLANNER", default=False):
        llm_plan = plan_with_llm(query=query, toolbox=toolbox, route=route)
        if llm_plan and llm_plan.calls:
            return complete_plan(query=query, route=route, plan=llm_plan)
    return AgentPlan(rationale="RAG planner selected semantic document retrieval.", calls=[_search_call(query, route)])


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


def compose_with_llm(query: str, plan: AgentPlan, evidence: Sequence[RetrievalResult]) -> str | None:
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
                    "retrieved_evidence": [evidence_to_dict(item) for item in evidence],
                },
                ensure_ascii=False,
            ),
        },
    ]
    return chat_with_configured_llm(messages, temperature=0.0)


def compose_from_evidence(query: str, evidence: Sequence[RetrievalResult]) -> str:
    if _is_unsupported_subjective_query(query):
        return (
            "The internal ECU documentation does not contain enough evidence about subjective appearance, "
            "industrial design, color, or visual preference to answer this question."
        )
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


def apply_grounding_checks(
    *,
    query: str,
    answer: str,
    sources: list[str],
    tool_results: Sequence[ToolResult],
    evidence: Sequence[RetrievalResult],
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

    if _is_unsupported_subjective_query(query):
        confidence = min(confidence, 0.45)
    if requested_models_without_evidence(query, evidence):
        confidence = min(confidence, 0.45)
    if sources and "source:" not in answer.lower() and "source" not in answer.lower():
        answer = f"{answer}\nsource: {', '.join(sources)}"
    return answer, confidence


def evidence_from_tool_results(tool_results: Sequence[ToolResult]) -> list[RetrievalResult]:
    evidence: list[RetrievalResult] = []
    for result in tool_results:
        evidence.extend(retrieval_results_from_tool(result))
    return _dedupe_evidence(evidence)


def evidence_to_dict(item: RetrievalResult) -> dict[str, Any]:
    return {
        "content": item.content,
        "metadata": item.metadata,
        "score": item.score,
    }


def models_in_query(query: str) -> list[str]:
    models = []
    for match in re.findall(r"(?<![A-Za-z0-9])ecu[-\s]?(\d+[a-z]?)(?![A-Za-z0-9])", query, flags=re.IGNORECASE):
        model = f"ECU-{match.upper()}"
        models.append(MODEL_ALIASES.get(model.upper(), _canonical_model_case(model)))
    return _dedupe(models)


def requested_models_without_evidence(query: str, evidence: Sequence[RetrievalResult]) -> list[str]:
    requested = models_in_query(query)
    if not requested:
        return []
    available = {item.metadata.get("model") for item in evidence}
    return [model for model in requested if model not in available]


def _search_call(query: str, route: Mapping[str, Any] | None) -> ToolCall:
    models = list((route or {}).get("models") or ())
    return ToolCall("search_documents", {"query": query, "models": models, "top_k": 10})


def _canonical_model_case(model: str) -> str:
    prefix, _, suffix = model.partition("-")
    if suffix.lower().endswith("b"):
        return f"{prefix.upper()}-{suffix[:-1]}b"
    return model.upper()


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


def _is_unsupported_subjective_query(query: str) -> bool:
    lowered = query.lower()
    subjective_terms = (
        "look better",
        "better looking",
        "appearance",
        "aesthetic",
        "visual",
        "color",
        "industrial design",
        "好看",
        "漂亮",
        "美观",
        "外观",
        "颜值",
        "颜色",
        "造型",
    )
    return any(term in lowered for term in subjective_terms)


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
