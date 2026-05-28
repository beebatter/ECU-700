from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from me_engineering_assistant.graph import ECUAgent


SERVER_NAME = "me-ecu-engineering-assistant"


def create_mcp_server(
    docs_dir: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError as exc:
        raise RuntimeError("Install the MCP extra first: python -m pip install -e '.[mcp]'") from exc

    agent = ECUAgent(docs_dir=docs_dir)
    toolbox = agent.toolbox
    documents_by_source = {document.metadata["source"]: document.text for document in agent.documents}
    server = FastMCP(
        SERVER_NAME,
        instructions=(
            "Ground ECU engineering answers in the internal ECU Markdown manuals. "
            "Use semantic document search before answering."
        ),
        host=host,
        port=port,
        json_response=True,
    )

    @server.tool()
    # pylint: disable=too-many-positional-arguments
    def search_documents(
        query: str,
        models: list[str] | None = None,
        series: list[str] | None = None,
        field: str | None = None,
        fields: list[str] | None = None,
        sources: list[str] | None = None,
        top_k: int = 6,
    ) -> dict[str, Any]:
        """Search internal ECU documentation chunks for grounding evidence."""
        if top_k < 1 or top_k > 10:
            raise ToolError("top_k must be between 1 and 10")
        return toolbox.search_documents(
            query=query,
            models=models,
            series=series,
            field=field,
            fields=fields,
            sources=sources,
            top_k=top_k,
        ).to_dict()

    @server.tool()
    def list_sources() -> dict[str, Any]:
        """List source documents available in the indexed internal documentation."""
        return toolbox.list_sources().to_dict()

    @server.tool()
    def get_document_catalog() -> dict[str, Any]:
        """Return the indexed ECU document catalog."""
        return toolbox.get_document_catalog().to_dict()

    @server.tool()
    def get_model_field_evidence(models: list[str] | None = None, field: str | None = None) -> dict[str, Any]:
        """Return structured model-field evidence extracted from source documents."""
        return toolbox.get_model_field_evidence(models=models, field=field).to_dict()

    @server.tool()
    def check_evidence_coverage(models: list[str], field: str) -> dict[str, Any]:
        """Check whether each requested model has structured evidence for a field."""
        return toolbox.check_evidence_coverage(models=models, field=field).to_dict()

    @server.resource("ecu://sources")
    def sources_resource() -> str:
        """Return available ECU source documents as JSON."""
        return json.dumps(toolbox.list_sources().to_dict(), indent=2)

    @server.resource("ecu://docs/{source}")
    def document_resource(source: str) -> str:
        """Return the raw Markdown for one ECU source document."""
        if source not in documents_by_source:
            available = ", ".join(sorted(documents_by_source))
            raise ToolError(f"Unknown source document: {source}. Available: {available}")
        return documents_by_source[source]

    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ME ECU assistant as a real MCP server.")
    parser.add_argument("--docs-dir", default=None, help="Directory containing ECU Markdown documents.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default="stdio",
        help="MCP transport to run. Use stdio for local clients and streamable-http for HTTP clients.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports.")
    args = parser.parse_args(argv)

    server = create_mcp_server(
        docs_dir=args.docs_dir,
        host=args.host,
        port=args.port,
    )
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
