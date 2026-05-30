from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping
from uuid import uuid4

from me_engineering_assistant.config import bool_env, env


DEFAULT_LOG_PATH = "logs/agent_events.jsonl"


def log_agent_response(
    *,
    query: str,
    response: Mapping[str, Any],
    trace: list[dict[str, Any]],
    latency_seconds: float,
    retriever_backend: str,
    docs_dir: str | None,
) -> dict[str, Any] | None:
    if not bool_env("ME_AGENT_LOG_ENABLED", default=True):
        return None

    record = build_agent_log_record(
        query=query,
        response=response,
        trace=trace,
        latency_seconds=latency_seconds,
        retriever_backend=retriever_backend,
        docs_dir=docs_dir,
    )
    path = Path(env("ME_AGENT_LOG_PATH", DEFAULT_LOG_PATH) or DEFAULT_LOG_PATH).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return None
    return record


def build_agent_log_record(
    *,
    query: str,
    response: Mapping[str, Any],
    trace: list[dict[str, Any]],
    latency_seconds: float,
    retriever_backend: str,
    docs_dir: str | None,
) -> dict[str, Any]:
    include_trace = bool_env("ME_AGENT_LOG_INCLUDE_TRACE", default=True)
    record = {
        "schema_version": "1.0",
        "event_type": "agent_response",
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_id": f"request-{uuid4().hex[:12]}",
        "query": query,
        "answer": response.get("answer"),
        "sources": list(response.get("sources") or ()),
        "confidence": response.get("confidence"),
        "route": response.get("route"),
        "needs_review": response.get("needs_review", False),
        "review_id": response.get("review_id"),
        "review_reason": response.get("review_reason"),
        "latency_seconds": latency_seconds,
        "llm_calls": response.get("llm_calls", 0),
        "llm_latency_seconds": response.get("llm_latency_seconds", 0.0),
        "retrieval_latency_seconds": response.get("retrieval_latency_seconds", 0.0),
        "generation_latency_seconds": response.get("generation_latency_seconds", 0.0),
        "retriever_backend": retriever_backend,
        "docs_dir": docs_dir,
    }
    if include_trace:
        record["trace"] = trace
    record["detectors"] = detect_log_record(record)
    return record


def detect_log_record(record: Mapping[str, Any]) -> dict[str, bool]:
    confidence = _float(record.get("confidence"), default=0.0)
    latency = _float(record.get("latency_seconds"), default=0.0)
    threshold = _float(env("ME_AGENT_LOG_LOW_CONFIDENCE_THRESHOLD", env("ME_HITL_CONFIDENCE_THRESHOLD", "0.75")), 0.75)
    slow_seconds = _float(env("ME_AGENT_LOG_SLOW_SECONDS", "10"), 10.0)
    return {
        "low_confidence": confidence < threshold,
        "missing_sources": not bool(record.get("sources")),
        "needs_review": bool(record.get("needs_review")),
        "slow_response": latency > slow_seconds,
    }


def load_log_records(path: str | Path | None = None) -> list[dict[str, Any]]:
    log_path = Path(path or env("ME_AGENT_LOG_PATH", DEFAULT_LOG_PATH) or DEFAULT_LOG_PATH).expanduser()
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"event_type": "parse_error", "raw": line})
    return records


def summarize_log_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [record for record in records if record.get("event_type") == "agent_response"]
    latencies = [_float(record.get("latency_seconds"), 0.0) for record in rows]
    llm_latencies = [_float(record.get("llm_latency_seconds"), 0.0) for record in rows]
    llm_calls = [_float(record.get("llm_calls"), 0.0) for record in rows]
    detectors = [record.get("detectors") or detect_log_record(record) for record in rows]
    return {
        "total": len(rows),
        "low_confidence_count": sum(1 for item in detectors if item.get("low_confidence")),
        "missing_sources_count": sum(1 for item in detectors if item.get("missing_sources")),
        "needs_review_count": sum(1 for item in detectors if item.get("needs_review")),
        "slow_response_count": sum(1 for item in detectors if item.get("slow_response")),
        "average_latency_seconds": mean(latencies) if latencies else 0.0,
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "average_llm_latency_seconds": mean(llm_latencies) if llm_latencies else 0.0,
        "average_llm_calls": mean(llm_calls) if llm_calls else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect structured ECU assistant logs.")
    parser.add_argument("--path", default=None, help="Path to JSONL log file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    records = load_log_records(args.path)
    summary = summarize_log_records(records)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("ECU Assistant Log Summary")
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
