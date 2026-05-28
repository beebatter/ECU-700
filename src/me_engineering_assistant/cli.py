from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys

from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.visualization import trace_to_json, trace_to_markdown, trace_to_mermaid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the ME ECU engineering assistant a question.")
    parser.add_argument("query", nargs="*", help="Engineering question to answer. Omit it to start interactive mode.")
    parser.add_argument("--docs-dir", default=None, help="Directory containing ECU markdown documents.")
    parser.add_argument("--session-id", default="default", help="Conversation session id for short and mid-term memory.")
    parser.add_argument("--memory-db", default=None, help="SQLite path for persistent conversation memory.")
    parser.add_argument("--memory-scope", default="global", help="Memory scope used to isolate projects or users.")
    parser.add_argument("--no-memory", action="store_true", help="Disable conversation memory for this run.")
    parser.add_argument("--trace", action="store_true", help="Show the agent route/retrieval/plan/tool/validation trace.")
    parser.add_argument(
        "--trace-format",
        choices=("json", "markdown", "mermaid"),
        default="markdown",
        help="Visualization format when --trace is enabled.",
    )
    parser.add_argument("--trace-file", default=None, help="Optional path to write the trace visualization.")
    args = parser.parse_args(argv)

    configure_cli_environment()
    agent = build_agent(
        docs_dir=args.docs_dir,
        memory_enabled=not args.no_memory,
        memory_path=args.memory_db,
        memory_scope=args.memory_scope,
    )
    if not args.query:
        return interactive_loop(agent, session_id=args.session_id)

    response = agent.answer(
        " ".join(args.query),
        include_trace=args.trace or bool(args.trace_file),
        session_id=args.session_id,
    )
    payload = response.to_dict()
    print(format_response(payload))
    if args.trace or args.trace_file:
        rendered = render_trace(payload, args.trace_format)
        if args.trace_file:
            with open(args.trace_file, "w", encoding="utf-8") as handle:
                handle.write(rendered)
        if args.trace:
            print()
            print(rendered, end="")
    return 0


def interactive_loop(agent: ECUAgent, session_id: str = "default") -> int:
    print("\n" + "=" * 72)
    print("ECU Engineering Assistant")
    print("=" * 72)
    print(f"Session: {session_id}")
    print("Type your question below. Type 'exit' or 'quit' to leave.")
    print("-" * 72)
    while True:
        try:
            query = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            return 0
        response = agent.answer(query, session_id=session_id)
        print()
        print(format_response(response.to_dict()))


def format_response(payload: dict) -> str:
    lines = [
        "",
        "=" * 72,
        "Answer",
        "-" * 72,
        str(payload.get("answer", "")).strip(),
    ]
    sources = payload.get("sources") or []
    if sources:
        lines.extend(["", "Sources", "-" * 72])
        lines.extend(f"- {source}" for source in sources)
    confidence = payload.get("confidence")
    if confidence is not None:
        lines.extend(["", "Confidence", "-" * 72, f"{float(confidence):.2f}"])
    if payload.get("needs_review"):
        reason = payload.get("review_reason") or "low confidence"
        lines.extend(["", "Review", "-" * 72, f"Required: {reason}"])
    lines.append("=" * 72)
    return "\n".join(lines)


def build_agent(
    docs_dir: str | None = None,
    *,
    memory_enabled: bool = True,
    memory_path: str | None = None,
    memory_scope: str = "global",
) -> ECUAgent:
    if os.environ.get("ME_CLI_VERBOSE_STARTUP", "").lower() in {"1", "true", "yes", "on"}:
        return ECUAgent(
            docs_dir=docs_dir,
            memory_enabled=memory_enabled,
            memory_path=memory_path,
            memory_scope=memory_scope,
        )

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with quiet_startup_output(captured_stdout, captured_stderr):
            return ECUAgent(
                docs_dir=docs_dir,
                memory_enabled=memory_enabled,
                memory_path=memory_path,
                memory_scope=memory_scope,
            )
    except Exception:
        sys.stdout.write(captured_stdout.getvalue())
        sys.stderr.write(captured_stderr.getvalue())
        raise


def configure_cli_environment() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import disable_progress_bars
    except ImportError:
        return
    disable_progress_bars()


@contextlib.contextmanager
def quiet_startup_output(stdout_buffer: io.StringIO, stderr_buffer: io.StringIO):
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


def render_trace(payload: dict, trace_format: str) -> str:
    if trace_format == "json":
        return trace_to_json(payload) + "\n"
    if trace_format == "mermaid":
        return trace_to_mermaid(payload.get("trace") or []) + "\n"
    return trace_to_markdown(payload)


if __name__ == "__main__":
    raise SystemExit(main())
