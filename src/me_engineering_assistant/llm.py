from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from me_engineering_assistant.config import bool_env, env
from me_engineering_assistant.retriever import RetrievalResult


@dataclass(frozen=True)
class LLMCallStats:
    count: int = 0
    latency_seconds: float = 0.0
    failures: int = 0
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["providers"] = list(self.providers)
        return payload


_LLM_CALL_STATS: ContextVar[LLMCallStats] = ContextVar("llm_call_stats", default=LLMCallStats())


def reset_llm_call_stats() -> None:
    _LLM_CALL_STATS.set(LLMCallStats())


def get_llm_call_stats() -> LLMCallStats:
    return _LLM_CALL_STATS.get()


def generate_with_configured_llm(query: str, retrieved: Sequence[RetrievalResult]) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an engineering assistant. Answer only from the supplied ECU "
                "documentation context. Return concise technical answers with source-aware wording."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{_format_context(retrieved)}\n\nQuestion: {query}",
        },
    ]
    return chat_with_configured_llm(messages, temperature=0.0)


def chat_with_configured_llm(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float = 0.0,
) -> str | None:
    if env("DATABRICKS_LLM_ENDPOINT"):
        databricks_answer = _record_llm_call(
            "databricks",
            lambda: chat_with_databricks(messages, temperature=temperature),
        )
        if databricks_answer:
            return databricks_answer

    if env("DEEPSEEK_API_KEY"):
        deepseek_answer = _record_llm_call(
            "deepseek",
            lambda: chat_with_deepseek(messages, temperature=temperature),
        )
        if deepseek_answer:
            return deepseek_answer
    return None


def json_with_configured_llm(messages: Sequence[Mapping[str, str]]) -> Any | None:
    content = chat_with_configured_llm(messages, temperature=0.0)
    if not content:
        return None
    try:
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return None


def chat_with_deepseek(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float = 0.0,
) -> str | None:
    api_key = env("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    base_url = (env("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "").rstrip("/")
    model = env("DEEPSEEK_MODEL", "deepseek-v4-flash")
    timeout = float(env("DEEPSEEK_TIMEOUT_SECONDS", "30") or "30")
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "stream": False,
    }
    if bool_env("DEEPSEEK_THINKING", default=False):
        payload["thinking"] = {"type": "enabled"}

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip() or None


def generate_with_deepseek_llm(query: str, retrieved: Sequence[RetrievalResult]) -> str | None:
    return generate_with_configured_llm(query, retrieved)


def chat_with_databricks(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float = 0.0,
) -> str | None:
    endpoint = env("DATABRICKS_LLM_ENDPOINT")
    if not endpoint:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_databricks import ChatDatabricks
    except ImportError:
        return None

    langchain_messages = []
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "system":
            langchain_messages.append(SystemMessage(content=content))
        else:
            langchain_messages.append(HumanMessage(content=content))

    try:
        response = ChatDatabricks(endpoint=endpoint, temperature=temperature).invoke(langchain_messages)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return getattr(response, "content", str(response)).strip()


def generate_with_databricks_llm(query: str, retrieved: Sequence[RetrievalResult]) -> str | None:
    return generate_with_configured_llm(query, retrieved)


def _format_context(retrieved: Sequence[RetrievalResult]) -> str:
    return "\n\n".join(
        f"Source: {result.metadata.get('source', 'unknown')}\n{result.content}" for result in retrieved
    )


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    first = min((index for index in (stripped.find("{"), stripped.find("[")) if index >= 0), default=0)
    last_object = stripped.rfind("}")
    last_array = stripped.rfind("]")
    last = max(last_object, last_array)
    return stripped[first : last + 1] if last >= first else stripped


def _record_llm_call(provider: str, call) -> str | None:
    started = time.perf_counter()
    success = False
    try:
        result = call()
        success = bool(result)
        return result
    finally:
        elapsed = time.perf_counter() - started
        current = _LLM_CALL_STATS.get()
        _LLM_CALL_STATS.set(
            LLMCallStats(
                count=current.count + 1,
                latency_seconds=current.latency_seconds + elapsed,
                failures=current.failures + (0 if success else 1),
                providers=current.providers + (provider,),
            )
        )
