from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

from me_engineering_assistant.documents import ModelFieldEvidence
from me_engineering_assistant.planner import QueryPlan
from me_engineering_assistant.retriever import RetrievalResult


@dataclass(frozen=True)
class CoverageItem:
    entity: str
    attribute: str
    covered: bool
    evidence: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageReport:
    complete: bool
    attribute: str | None
    items: list[CoverageItem]
    missing: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "attribute": self.attribute,
            "items": [item.to_dict() for item in self.items],
            "missing": self.missing,
        }


def check_plan_coverage(plan: QueryPlan, evidence: Sequence[RetrievalResult]) -> CoverageReport:
    if not plan.entities or not plan.attribute:
        return CoverageReport(complete=bool(evidence), attribute=plan.attribute, items=[], missing=[])

    items = []
    missing = []
    for entity in plan.entities:
        rows = [_evidence_dict(result) for result in evidence if _matches_entity_field(result, entity, plan.attribute)]
        covered = bool(rows)
        items.append(CoverageItem(entity=entity, attribute=plan.attribute, covered=covered, evidence=rows))
        if not covered:
            missing.append({"entity": entity, "attribute": plan.attribute})
    return CoverageReport(
        complete=not missing,
        attribute=plan.attribute,
        items=items,
        missing=missing,
    )


def model_field_evidence(
    rows: Sequence[ModelFieldEvidence],
    *,
    models: Sequence[str] | None = None,
    field: str | None = None,
) -> list[dict[str, str]]:
    model_set = set(models or ())
    selected = []
    for row in rows:
        if model_set and row.model not in model_set:
            continue
        if field and not field_matches(field, row.field):
            continue
        selected.append(row.to_dict())
    return selected


def field_matches(requested: str, actual: str) -> bool:
    if not requested:
        return True
    requested_slug = _field_slug(requested)
    actual_slug = _field_slug(actual)
    if requested_slug == actual_slug:
        return True
    requested_tokens = _field_tokens(requested_slug)
    actual_tokens = _field_tokens(actual_slug)
    if not requested_tokens or not actual_tokens:
        return False
    return requested_tokens <= actual_tokens or actual_tokens <= requested_tokens or bool(requested_tokens & actual_tokens)


def _matches_entity_field(result: RetrievalResult, entity: str, attribute: str) -> bool:
    metadata = result.metadata
    if metadata.get("model") != entity:
        return False
    if field_matches(attribute, metadata.get("field", "")):
        return True
    if field_matches(attribute, metadata.get("field_label", "")):
        return True
    content_tokens = _field_tokens(result.content)
    return bool(_field_tokens(attribute) & content_tokens)


def _evidence_dict(result: RetrievalResult) -> dict[str, str]:
    metadata = result.metadata
    return {
        "model": metadata.get("model", ""),
        "series": metadata.get("series", ""),
        "field": metadata.get("field", ""),
        "field_label": metadata.get("field_label", ""),
        "value": metadata.get("value", ""),
        "source": metadata.get("source", ""),
        "section": metadata.get("section", ""),
        "content": result.content,
    }


def _field_slug(value: str) -> str:
    return "_".join(_field_tokens(value))


def _field_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))
