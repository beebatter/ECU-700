from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from me_engineering_assistant.config import bool_env, env
from me_engineering_assistant.retriever import RetrievalResult


@dataclass(frozen=True)
class ReviewDecision:
    needs_review: bool
    review_id: str | None = None
    reason: str | None = None


def maybe_enqueue_review(
    *,
    query: str,
    answer: str,
    confidence: float,
    sources: list[str],
    route: dict[str, object],
    retrieved: list[RetrievalResult],
) -> ReviewDecision:
    if not bool_env("ME_HITL_ENABLED", default=True):
        return ReviewDecision(needs_review=False)

    threshold = float(env("ME_HITL_CONFIDENCE_THRESHOLD", "0.75") or "0.75")
    if confidence >= threshold:
        return ReviewDecision(needs_review=False)

    reason = f"confidence {confidence:.2f} below threshold {threshold:.2f}"
    review_id = _write_review_case(
        query=query,
        answer=answer,
        confidence=confidence,
        sources=sources,
        route=route,
        retrieved=retrieved,
        reason=reason,
    )
    return ReviewDecision(needs_review=True, review_id=review_id, reason=reason)


def _write_review_case(
    *,
    query: str,
    answer: str,
    confidence: float,
    sources: list[str],
    route: dict[str, object],
    retrieved: list[RetrievalResult],
    reason: str,
) -> str:
    review_id = f"review-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    record = {
        "review_id": review_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "query": query,
        "draft_answer": answer,
        "confidence": confidence,
        "sources": sources,
        "route": route,
        "retrieved_context": [
            {
                "source": result.metadata.get("source"),
                "model": result.metadata.get("model"),
                "series": result.metadata.get("series"),
                "section": result.metadata.get("section"),
                "score": result.score,
                "excerpt": result.content[:500],
            }
            for result in retrieved
        ],
    }
    path = Path(env("ME_REVIEW_QUEUE_PATH", "review_queue.jsonl") or "review_queue.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return review_id

