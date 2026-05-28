from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from me_engineering_assistant.config import bool_env, env
from me_engineering_assistant.documents import CatalogEntry, ModelFieldEvidence
from me_engineering_assistant.llm import json_with_configured_llm


ALL_MODELS = ("ECU-750", "ECU-850", "ECU-850b")
DEFAULT_SERIES_TO_MODELS = {
    "ECU-700": ("ECU-750",),
    "ECU-800": ("ECU-850", "ECU-850b"),
}


@dataclass(frozen=True)
class QueryPlan:
    task_type: str
    entities: list[str]
    attribute: str | None
    scope: str
    route: str
    subqueries: list[dict[str, str]]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_query(
    query: str,
    *,
    llm_enabled: bool | None = None,
    catalog: Sequence[CatalogEntry] | None = None,
    field_table: Sequence[ModelFieldEvidence] | None = None,
) -> QueryPlan:
    """
    Build a retrieval plan without hard-coded user-intent branches.

    If an LLM is configured, it may produce a structured retrieval plan from the
    document catalog and dynamically extracted fields. The local fallback only
    links explicit ECU entities to metadata and otherwise leaves the query broad
    so retrieval, coverage, and grounding operate from evidence instead of
    pre-enumerated question types.
    """

    base_plan = semantic_fallback_plan(query, catalog=catalog)
    use_llm = bool_env("ME_USE_LLM_PLANNER", default=bool(env("DEEPSEEK_API_KEY"))) if llm_enabled is None else llm_enabled
    if not use_llm:
        return base_plan

    llm_plan = plan_with_llm(query=query, base_plan=base_plan, catalog=catalog, field_table=field_table)
    if llm_plan is None:
        return base_plan
    return sanitize_query_plan(llm_plan, query=query, catalog=catalog, field_table=field_table)


def semantic_fallback_plan(
    query: str,
    *,
    catalog: Sequence[CatalogEntry] | None = None,
) -> QueryPlan:
    known_models = models_from_catalog(catalog)
    entities = [
        model
        for model in models_in_query(query, catalog=catalog, include_unknown=False)
        if model in known_models
    ]
    scope = scope_from_entities(entities, known_models)
    return QueryPlan(
        task_type="rag",
        entities=entities,
        attribute=None,
        scope=scope,
        route="standard",
        subqueries=build_subqueries(query=query, entities=entities, attribute=None),
        reasons=["explicit_entity_linking"] if entities else ["semantic_retrieval"],
    )


def plan_with_llm(
    *,
    query: str,
    base_plan: QueryPlan,
    catalog: Sequence[CatalogEntry] | None = None,
    field_table: Sequence[ModelFieldEvidence] | None = None,
) -> QueryPlan | None:
    payload = json_with_configured_llm(
        [
            {
                "role": "system",
                "content": (
                    "You create retrieval plans for an ECU documentation RAG agent. "
                    "Return strict JSON only. Do not answer the user. Do not invent "
                    "facts. Use only the provided document catalog and field catalog. "
                    "The plan is for retrieval, not final reasoning. Keep task_type, "
                    "scope, and route as short descriptive labels rather than forcing "
                    "the question into predefined categories."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "document_catalog": [entry.to_dict() for entry in catalog or ()],
                        "field_catalog": field_catalog(field_table),
                        "base_plan": base_plan.to_dict(),
                        "required_json_shape": {
                            "task_type": "short descriptive label",
                            "entities": ["known ECU model identifiers only"],
                            "attribute": "one field slug from field_catalog, or null",
                            "scope": "short descriptive label",
                            "route": "short descriptive retrieval strategy label",
                            "subqueries": [
                                {
                                    "entity": "optional ECU model",
                                    "attribute": "optional field slug",
                                    "query": "specific retrieval query",
                                }
                            ],
                            "reasons": ["brief evidence-planning reason"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    )
    if not isinstance(payload, dict):
        return None

    return QueryPlan(
        task_type=str(payload.get("task_type") or base_plan.task_type),
        entities=[str(item) for item in payload.get("entities") or base_plan.entities],
        attribute=str(payload["attribute"]) if payload.get("attribute") else base_plan.attribute,
        scope=str(payload.get("scope") or base_plan.scope),
        route=str(payload.get("route") or base_plan.route),
        subqueries=[
            {str(key): str(value) for key, value in item.items()}
            for item in payload.get("subqueries", [])
            if isinstance(item, dict)
        ]
        or base_plan.subqueries,
        reasons=[str(item) for item in payload.get("reasons") or base_plan.reasons],
    )


def sanitize_query_plan(
    plan: QueryPlan,
    *,
    query: str = "",
    catalog: Sequence[CatalogEntry] | None = None,
    field_table: Sequence[ModelFieldEvidence] | None = None,
) -> QueryPlan:
    known_models = models_from_catalog(catalog)
    known_fields = fields_from_table(field_table)
    entities = _dedupe([entity for entity in plan.entities if entity in known_models])
    attribute = _sanitize_attribute(plan.attribute, known_fields=known_fields)
    subqueries = [
        {
            "entity": str(item.get("entity", "")),
            "attribute": str(item.get("attribute", attribute or "")),
            "query": str(item.get("query", "")),
        }
        for item in plan.subqueries
        if item.get("query")
    ]
    if not subqueries:
        subqueries = build_subqueries(query=query, entities=entities, attribute=attribute)

    return QueryPlan(
        task_type=_sanitize_label(plan.task_type) or "rag",
        entities=entities,
        attribute=attribute,
        scope=_sanitize_label(plan.scope) or scope_from_entities(entities, known_models),
        route=_sanitize_label(plan.route) or "standard",
        subqueries=subqueries,
        reasons=list(plan.reasons or ["sanitized_plan"]),
    )


def build_subqueries(query: str, entities: Sequence[str], attribute: str | None) -> list[dict[str, str]]:
    if not entities:
        return [{"entity": "", "attribute": attribute or "", "query": query}]

    subqueries = []
    for entity in entities:
        subquery = f"{query} {entity}".strip()
        if attribute:
            subquery = f"{subquery} {attribute.replace('_', ' ')}".strip()
        subqueries.append({"entity": entity, "attribute": attribute or "", "query": subquery})
    return subqueries


def models_in_query(
    query: str,
    *,
    catalog: Sequence[CatalogEntry] | None = None,
    include_unknown: bool = True,
) -> list[str]:
    known_models = models_from_catalog(catalog)
    series_to_models = series_models_from_catalog(catalog)
    models: list[str] = []

    for model in known_models:
        if _identifier_in_query(model, query):
            models.append(model)

    for series, series_models in series_to_models.items():
        if _identifier_in_query(series, query):
            models.extend(series_models)

    if include_unknown:
        for match in re.findall(r"(?<![A-Za-z0-9])ecu[-\s]?(\d+[a-z]?)(?![A-Za-z0-9])", query, flags=re.IGNORECASE):
            model = _canonical_model_case(f"ECU-{match.upper()}")
            if model not in known_models:
                models.append(model)

    return _dedupe(models)


def models_from_catalog(catalog: Sequence[CatalogEntry] | None) -> list[str]:
    models = [entry.model for entry in catalog or () if entry.model and entry.model != "unknown"]
    return _dedupe(models) or list(ALL_MODELS)


def series_models_from_catalog(catalog: Sequence[CatalogEntry] | None) -> dict[str, list[str]]:
    if not catalog:
        return {series: list(models) for series, models in DEFAULT_SERIES_TO_MODELS.items()}

    by_series: dict[str, list[str]] = {}
    for entry in catalog:
        if not entry.series or entry.series == "unknown" or not entry.model or entry.model == "unknown":
            continue
        by_series.setdefault(entry.series, [])
        if entry.model not in by_series[entry.series]:
            by_series[entry.series].append(entry.model)
    return by_series


def fields_from_table(field_table: Sequence[ModelFieldEvidence] | None) -> list[str]:
    fields = [row.field for row in field_table or () if row.field]
    return _dedupe(fields)


def field_catalog(field_table: Sequence[ModelFieldEvidence] | None) -> list[dict[str, str]]:
    rows = []
    seen: set[str] = set()
    for row in field_table or ():
        if not row.field or row.field in seen:
            continue
        seen.add(row.field)
        rows.append({"field": row.field, "label": row.field_label})
    return rows


def scope_from_entities(entities: Sequence[str], known_models: Sequence[str]) -> str:
    if not entities:
        return "semantic"
    if set(entities) == set(known_models):
        return "all_indexed_models"
    if len(entities) == 1:
        return "model_filtered"
    return "multi_model_filtered"


def _sanitize_attribute(attribute: str | None, *, known_fields: Sequence[str]) -> str | None:
    if not attribute:
        return None
    cleaned = _sanitize_label(attribute)
    if not cleaned:
        return None
    if not known_fields:
        return cleaned

    for field in known_fields:
        if cleaned == field or _token_overlap(cleaned, field):
            return field
    return cleaned


def _sanitize_label(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def _token_overlap(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    return bool(left_tokens and right_tokens and left_tokens & right_tokens)


def _canonical_model_case(model: str) -> str:
    prefix, _, suffix = model.partition("-")
    if suffix.lower().endswith("b"):
        return f"{prefix.upper()}-{suffix[:-1]}b"
    return model.upper()


def _identifier_in_query(identifier: str, query: str) -> bool:
    parts = re.findall(r"[a-z]+|\d+[a-z]?", identifier.lower())
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"[-\s]?".join(re.escape(part) for part in parts) + r"(?![a-z0-9])"
    return bool(re.search(pattern, query.lower()))


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
