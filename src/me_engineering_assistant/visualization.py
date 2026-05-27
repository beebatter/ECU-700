from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def trace_step(name: str, status: str = "completed", **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def trace_to_markdown(response: Mapping[str, Any]) -> str:
    trace = list(response.get("trace") or ())
    lines = ["# ECU Assistant Trace", ""]
    answer = response.get("answer")
    if answer:
        lines.extend(["## Final Answer", "", str(answer), ""])
    lines.extend(["## Steps", ""])
    for index, step in enumerate(trace, start=1):
        details = dict(step.get("details") or {})
        lines.append(f"{index}. **{step.get('name', 'unknown')}** - {step.get('status', 'unknown')}")
        summary = details.pop("summary", None)
        if summary:
            lines.append(f"   - summary: {summary}")
        for key, value in details.items():
            lines.append(f"   - {key}: `{_compact(value)}`")
    if trace:
        lines.extend(["", "## Mermaid", "", "```mermaid", trace_to_mermaid(trace), "```"])
    return "\n".join(lines).rstrip() + "\n"


def trace_to_mermaid(trace: Sequence[Mapping[str, Any]]) -> str:
    lines = ["flowchart LR"]
    for index, step in enumerate(trace):
        node_id = f"S{index}"
        label = f"{index + 1}. {step.get('name', 'unknown')}\\n{step.get('status', 'unknown')}"
        lines.append(f'  {node_id}["{_escape_mermaid(label)}"]')
        if index:
            lines.append(f"  S{index - 1} --> {node_id}")
    return "\n".join(lines)


def trace_to_json(response: Mapping[str, Any]) -> str:
    return json.dumps(response, indent=2, ensure_ascii=False)


def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _escape_mermaid(value: str) -> str:
    return value.replace('"', "'")
