from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from me_engineering_assistant.config import env


DEFAULT_MEMORY_PATH = ".me_engineering_memory.sqlite"
GLOBAL_SCOPE = "global"


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    scope: str
    kind: str
    content: str
    importance: float
    metadata: dict[str, Any]
    created_at: float
    updated_at: float
    last_used_at: float | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope": self.scope,
            "kind": self.kind,
            "content": self.content,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "score": self.score,
        }


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: int
    scope: str
    session_id: str
    turn_index: int
    user_message: str
    assistant_answer: str
    confidence: float
    sources: list[str]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "scope": self.scope,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "assistant_answer": self.assistant_answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "created_at": self.created_at,
        }


class MemoryStore:
    """SQLite-backed memory store for sessions, summaries, and long-term facts."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or env("ME_MEMORY_DB_PATH", DEFAULT_MEMORY_PATH) or DEFAULT_MEMORY_PATH
        self.path = Path(configured_path).expanduser()
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def add_memory(
        self,
        content: str,
        *,
        scope: str = GLOBAL_SCOPE,
        kind: str = "long_term",
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        normalized = _compact_text(content)
        if not normalized:
            raise ValueError("memory content cannot be empty")

        now = time.time()
        memory_id = _memory_id(scope=scope, kind=kind, content=normalized)
        with self._connection() as connection:
            existing = connection.execute(
                "select created_at from memories where memory_id = ?",
                (memory_id,),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            connection.execute(
                """
                insert into memories (
                    memory_id, scope, kind, content, importance, metadata_json,
                    created_at, updated_at, last_used_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, null)
                on conflict(memory_id) do update set
                    content = excluded.content,
                    importance = max(memories.importance, excluded.importance),
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    scope,
                    kind,
                    normalized,
                    float(importance),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at,
                    now,
                ),
            )
        return MemoryRecord(
            memory_id=memory_id,
            scope=scope,
            kind=kind,
            content=normalized,
            importance=float(importance),
            metadata=metadata or {},
            created_at=created_at,
            updated_at=now,
        )

    def search_memories(
        self,
        query: str,
        *,
        scope: str = GLOBAL_SCOPE,
        kinds: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        placeholders = ",".join("?" for _ in (kinds or ()))
        where = "scope in (?, ?)"
        params: list[Any] = [scope, GLOBAL_SCOPE]
        if kinds:
            where += f" and kind in ({placeholders})"
            params.extend(kinds)

        with self._connection() as connection:
            rows = connection.execute(
                f"select * from memories where {where}",
                params,
            ).fetchall()

        scored: list[MemoryRecord] = []
        for row in rows:
            record = _record_from_row(row)
            overlap = len(query_tokens & _tokens(record.content))
            if overlap == 0:
                continue
            score = overlap + (record.importance * 0.2) + _recency_bonus(record.updated_at)
            scored.append(_with_score(record, score))

        scored.sort(key=lambda item: item.score, reverse=True)
        selected = scored[: max(0, limit)]
        self._mark_used([item.memory_id for item in selected])
        return selected

    def append_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_answer: str,
        confidence: float,
        sources: Sequence[str],
        scope: str = GLOBAL_SCOPE,
    ) -> ConversationTurn:
        now = time.time()
        with self._connection() as connection:
            row = connection.execute(
                "select coalesce(max(turn_index), -1) + 1 as next_index from turns where scope = ? and session_id = ?",
                (scope, session_id),
            ).fetchone()
            turn_index = int(row["next_index"])
            cursor = connection.execute(
                """
                insert into turns (
                    scope, session_id, turn_index, user_message, assistant_answer,
                    confidence, sources_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope,
                    session_id,
                    turn_index,
                    _compact_text(user_message, max_chars=4_000),
                    _compact_text(assistant_answer, max_chars=8_000),
                    float(confidence),
                    json.dumps(list(sources), ensure_ascii=False),
                    now,
                ),
            )
            turn_id = int(cursor.lastrowid)
        return ConversationTurn(
            turn_id=turn_id,
            scope=scope,
            session_id=session_id,
            turn_index=turn_index,
            user_message=user_message,
            assistant_answer=assistant_answer,
            confidence=float(confidence),
            sources=list(sources),
            created_at=now,
        )

    def recent_turns(
        self,
        *,
        session_id: str,
        scope: str = GLOBAL_SCOPE,
        limit: int = 6,
    ) -> list[ConversationTurn]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                select * from turns
                where scope = ? and session_id = ?
                order by turn_index desc
                limit ?
                """,
                (scope, session_id, max(0, limit)),
            ).fetchall()
        return [_turn_from_row(row) for row in reversed(rows)]

    def get_session_summary(self, *, session_id: str, scope: str = GLOBAL_SCOPE) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "select summary from session_summaries where scope = ? and session_id = ?",
                (scope, session_id),
            ).fetchone()
        return str(row["summary"]) if row else ""

    def upsert_session_summary(self, *, session_id: str, summary: str, scope: str = GLOBAL_SCOPE) -> None:
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                """
                insert into session_summaries (scope, session_id, summary, updated_at)
                values (?, ?, ?, ?)
                on conflict(scope, session_id) do update set
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (scope, session_id, _compact_text(summary, max_chars=2_000), now),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                create table if not exists memories (
                    memory_id text primary key,
                    scope text not null,
                    kind text not null,
                    content text not null,
                    importance real not null,
                    metadata_json text not null,
                    created_at real not null,
                    updated_at real not null,
                    last_used_at real
                );
                create index if not exists idx_memories_scope_kind
                    on memories(scope, kind, updated_at);

                create table if not exists turns (
                    turn_id integer primary key autoincrement,
                    scope text not null,
                    session_id text not null,
                    turn_index integer not null,
                    user_message text not null,
                    assistant_answer text not null,
                    confidence real not null,
                    sources_json text not null,
                    created_at real not null
                );
                create index if not exists idx_turns_session
                    on turns(scope, session_id, turn_index);

                create table if not exists session_summaries (
                    scope text not null,
                    session_id text not null,
                    summary text not null,
                    updated_at real not null,
                    primary key(scope, session_id)
                );
                """
            )

    def _mark_used(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connection() as connection:
            connection.execute(
                f"update memories set last_used_at = ? where memory_id in ({placeholders})",
                [time.time(), *memory_ids],
            )


def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=str(row["memory_id"]),
        scope=str(row["scope"]),
        kind=str(row["kind"]),
        content=str(row["content"]),
        importance=float(row["importance"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_used_at=float(row["last_used_at"]) if row["last_used_at"] is not None else None,
    )


def _turn_from_row(row: sqlite3.Row) -> ConversationTurn:
    return ConversationTurn(
        turn_id=int(row["turn_id"]),
        scope=str(row["scope"]),
        session_id=str(row["session_id"]),
        turn_index=int(row["turn_index"]),
        user_message=str(row["user_message"]),
        assistant_answer=str(row["assistant_answer"]),
        confidence=float(row["confidence"]),
        sources=list(json.loads(row["sources_json"] or "[]")),
        created_at=float(row["created_at"]),
    )


def _with_score(record: MemoryRecord, score: float) -> MemoryRecord:
    return MemoryRecord(
        memory_id=record.memory_id,
        scope=record.scope,
        kind=record.kind,
        content=record.content,
        importance=record.importance,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_used_at=record.last_used_at,
        score=score,
    )


def _memory_id(*, scope: str, kind: str, content: str) -> str:
    digest = hashlib.sha256(f"{scope}:{kind}:{content.lower()}".encode("utf-8")).hexdigest()
    return f"mem-{digest[:20]}"


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value.lower()))


def _compact_text(value: str, max_chars: int = 1_000) -> str:
    compacted = re.sub(r"\s+", " ", value).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


def _recency_bonus(updated_at: float) -> float:
    age_days = max(0.0, (time.time() - updated_at) / 86_400)
    return 0.1 / (1.0 + age_days)
