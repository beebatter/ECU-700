from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    kind: str
    importance: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReflectionResult:
    memory_candidates: list[MemoryCandidate]
    session_summary: str


def reflect_turn(
    *,
    query: str,
    answer: str,
    confidence: float,
    sources: Sequence[str],
    previous_summary: str,
) -> ReflectionResult:
    safe_query = redact_sensitive_text(query)
    safe_answer = redact_sensitive_text(answer)
    candidates = [] if _contains_sensitive_material(query) or _contains_sensitive_material(answer) else extract_memory_candidates(safe_query)
    return ReflectionResult(
        memory_candidates=candidates,
        session_summary=update_session_summary(
            previous_summary=previous_summary,
            query=safe_query,
            answer=safe_answer,
            confidence=confidence,
            sources=sources,
        ),
    )


def extract_memory_candidates(text: str) -> list[MemoryCandidate]:
    if _contains_sensitive_material(text):
        return []

    lowered = text.lower()
    if _contains_any(lowered, PREFERENCE_TRIGGERS):
        return [
            MemoryCandidate(
                content=f"User preference: {_compact_text(text)}",
                kind="preference",
                importance=0.9,
                metadata={"source": "reflection", "reason": "preference_trigger"},
            )
        ]
    if _contains_any(lowered, PROJECT_DECISION_TRIGGERS):
        return [
            MemoryCandidate(
                content=f"Project decision: {_compact_text(text)}",
                kind="project_decision",
                importance=0.8,
                metadata={"source": "reflection", "reason": "project_decision_trigger"},
            )
        ]
    return []


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED_SECRET]", redacted, flags=re.IGNORECASE)
    return redacted


def update_session_summary(
    *,
    previous_summary: str,
    query: str,
    answer: str,
    confidence: float,
    sources: Sequence[str],
    max_chars: int = 1_800,
) -> str:
    source_text = ", ".join(sources[:3]) if sources else "no sources"
    line = (
        f"- User asked: {_compact_text(query, 180)} | "
        f"Assistant confidence: {confidence:.2f}; sources: {source_text}; "
        f"answer gist: {_compact_text(answer, 220)}"
    )
    combined = "\n".join(part for part in (previous_summary.strip(), line) if part)
    if len(combined) <= max_chars:
        return combined

    trimmed = combined[-max_chars:]
    first_newline = trimmed.find("\n")
    return trimmed[first_newline + 1 :] if first_newline >= 0 else trimmed


PREFERENCE_TRIGGERS = (
    "remember",
    "please remember",
    "i prefer",
    "i want",
    "always",
    "never",
    "do not",
    "don't",
    "以后",
    "记住",
    "请记住",
    "不要",
    "不希望",
    "我希望",
    "我的想法",
    "偏好",
)

PROJECT_DECISION_TRIGGERS = (
    "we decided",
    "project should",
    "standard rag",
    "sentence-transformers",
    "faiss",
    "mcp",
    "改成",
    "只保留",
    "使用",
    "标准rag",
    "标准 rag",
)

SENSITIVE_PATTERNS = (
    r"\bsk-[a-z0-9]{8,}\b",
    r"\bapi[_ -]?key\s*[:=]\s*\S+",
    r"\bpassword\s*[:=]\s*\S+",
    r"\btoken\s*[:=]\s*\S+",
    r"密钥\s*[:：]\s*\S+",
    r"密码\s*[:：]\s*\S+",
)


def _contains_sensitive_material(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in SENSITIVE_PATTERNS)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _compact_text(value: str, max_chars: int = 500) -> str:
    compacted = re.sub(r"\s+", " ", value).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."
