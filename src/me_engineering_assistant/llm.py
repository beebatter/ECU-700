from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from me_engineering_assistant.config import bool_env, env
from me_engineering_assistant.retriever import RetrievalResult


def generate_with_configured_llm(query: str, retrieved: Sequence[RetrievalResult]) -> str | None:
    deepseek_answer = generate_with_deepseek_llm(query, retrieved)
    if deepseek_answer:
        return deepseek_answer
    return generate_with_databricks_llm(query, retrieved)


def chat_with_configured_llm(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float = 0.0,
) -> str | None:
    deepseek_answer = chat_with_deepseek(messages, temperature=temperature)
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
    return chat_with_deepseek(messages, temperature=0.0)


def generate_with_databricks_llm(query: str, retrieved: Sequence[RetrievalResult]) -> str | None:
    endpoint = os.getenv("DATABRICKS_LLM_ENDPOINT")
    if not endpoint:
        return None

    try:
        from langchain_databricks import ChatDatabricks
    except ImportError:
        return None

    prompt = (
        "You are an engineering assistant. Answer only from the ECU documentation context. "
        "If the answer is not present, say what is missing.\n\n"
        f"Context:\n{_format_context(retrieved)}\n\nQuestion: {query}\nAnswer:"
    )
    response = ChatDatabricks(endpoint=endpoint, temperature=0).invoke(prompt)
    return getattr(response, "content", str(response)).strip()


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
