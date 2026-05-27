from __future__ import annotations

import argparse
import json

from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.visualization import trace_to_json, trace_to_markdown, trace_to_mermaid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the ME ECU engineering assistant a question.")
    parser.add_argument("query", nargs="+", help="Engineering question to answer.")
    parser.add_argument("--docs-dir", default=None, help="Directory containing ECU markdown documents.")
    parser.add_argument("--trace", action="store_true", help="Show the agent route/retrieval/plan/tool/validation trace.")
    parser.add_argument(
        "--trace-format",
        choices=("json", "markdown", "mermaid"),
        default="markdown",
        help="Visualization format when --trace is enabled.",
    )
    parser.add_argument("--trace-file", default=None, help="Optional path to write the trace visualization.")
    args = parser.parse_args(argv)

    agent = ECUAgent(docs_dir=args.docs_dir)
    response = agent.answer(" ".join(args.query), include_trace=args.trace or bool(args.trace_file))
    payload = response.to_dict()
    if args.trace or args.trace_file:
        rendered = render_trace(payload, args.trace_format)
        if args.trace_file:
            with open(args.trace_file, "w", encoding="utf-8") as handle:
                handle.write(rendered)
        if args.trace:
            print(rendered, end="")
        else:
            print(json.dumps({key: value for key, value in payload.items() if key != "trace"}, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def render_trace(payload: dict, trace_format: str) -> str:
    if trace_format == "json":
        return trace_to_json(payload) + "\n"
    if trace_format == "mermaid":
        return trace_to_mermaid(payload.get("trace") or []) + "\n"
    return trace_to_markdown(payload)


if __name__ == "__main__":
    raise SystemExit(main())
