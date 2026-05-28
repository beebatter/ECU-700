from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from me_engineering_assistant.memory import ConversationTurn, GLOBAL_SCOPE, MemoryRecord, MemoryStore
from me_engineering_assistant.reflection import ReflectionResult, redact_sensitive_text, reflect_turn


DEFAULT_SESSION_ID = "default"


@dataclass(frozen=True)
class MemoryContext:
    session_id: str
    scope: str
    summary: str
    recent_turns: list[ConversationTurn]
    memories: list[MemoryRecord]

    def is_empty(self) -> bool:
        return not self.summary and not self.recent_turns and not self.memories

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scope": self.scope,
            "summary": self.summary,
            "recent_turns": [turn.to_dict() for turn in self.recent_turns],
            "memories": [memory.to_dict() for memory in self.memories],
        }

    def as_prompt_context(self) -> str:
        sections = []
        if self.summary:
            sections.append(f"Session summary:\n{self.summary}")
        if self.recent_turns:
            lines = []
            for turn in self.recent_turns:
                lines.append(f"User: {_compact(turn.user_message, 240)}")
                lines.append(f"Assistant: {_compact(turn.assistant_answer, 320)}")
            sections.append("Recent conversation:\n" + "\n".join(lines))
        if self.memories:
            lines = [f"- [{memory.kind}] {memory.content}" for memory in self.memories]
            sections.append("Relevant long-term memory:\n" + "\n".join(lines))
        return "\n\n".join(sections)


@dataclass(frozen=True)
class ReflectionWriteResult:
    turn: ConversationTurn
    reflection: ReflectionResult
    stored_memory_ids: list[str]


class ConversationManager:
    """Builds prompt context and records conversation memory around the agent."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        scope: str = GLOBAL_SCOPE,
        recent_turn_limit: int = 6,
        memory_limit: int = 5,
    ) -> None:
        self.store = store
        self.scope = scope
        self.recent_turn_limit = recent_turn_limit
        self.memory_limit = memory_limit

    def build_context(self, query: str, session_id: str | None = None) -> MemoryContext:
        resolved_session = session_id or DEFAULT_SESSION_ID
        summary = self.store.get_session_summary(session_id=resolved_session, scope=self.scope)
        recent_turns = self.store.recent_turns(
            session_id=resolved_session,
            scope=self.scope,
            limit=self.recent_turn_limit,
        )
        memory_query = _memory_search_text(query, summary, recent_turns)
        memories = self.store.search_memories(
            memory_query,
            scope=self.scope,
            kinds=("preference", "project_decision", "long_term"),
            limit=self.memory_limit,
        )
        return MemoryContext(
            session_id=resolved_session,
            scope=self.scope,
            summary=summary,
            recent_turns=recent_turns,
            memories=memories,
        )

    def enrich_query(self, query: str, context: MemoryContext) -> str:
        if context.is_empty():
            return query
        return f"{context.as_prompt_context()}\n\nCurrent question:\n{query}"

    def record_turn(
        self,
        *,
        session_id: str,
        query: str,
        answer: str,
        confidence: float,
        sources: Sequence[str],
        previous_summary: str,
    ) -> ReflectionWriteResult:
        safe_query = redact_sensitive_text(query)
        safe_answer = redact_sensitive_text(answer)
        turn = self.store.append_turn(
            session_id=session_id,
            scope=self.scope,
            user_message=safe_query,
            assistant_answer=safe_answer,
            confidence=confidence,
            sources=sources,
        )
        reflection = reflect_turn(
            query=query,
            answer=answer,
            confidence=confidence,
            sources=sources,
            previous_summary=previous_summary,
        )
        stored_ids = []
        for candidate in reflection.memory_candidates:
            memory = self.store.add_memory(
                candidate.content,
                scope=self.scope,
                kind=candidate.kind,
                importance=candidate.importance,
                metadata=candidate.metadata,
            )
            stored_ids.append(memory.memory_id)
        self.store.upsert_session_summary(
            session_id=session_id,
            scope=self.scope,
            summary=reflection.session_summary,
        )
        return ReflectionWriteResult(
            turn=turn,
            reflection=reflection,
            stored_memory_ids=stored_ids,
        )


def context_from_dict(payload: dict[str, Any] | None) -> MemoryContext | None:
    if not payload:
        return None
    return MemoryContext(
        session_id=str(payload.get("session_id") or DEFAULT_SESSION_ID),
        scope=str(payload.get("scope") or GLOBAL_SCOPE),
        summary=str(payload.get("summary") or ""),
        recent_turns=[],
        memories=[],
    )


def _memory_search_text(query: str, summary: str, recent_turns: Sequence[ConversationTurn]) -> str:
    recent_text = " ".join(turn.user_message for turn in recent_turns[-3:])
    return " ".join(part for part in (query, summary, recent_text) if part)


def _compact(value: str, max_chars: int) -> str:
    compacted = re.sub(r"\s+", " ", value).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."
