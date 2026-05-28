# ME Engineering Assistant Agent

This project implements the ECU engineering coding challenge as a small Python
package instead of a notebook. It provides a LangGraph-style RAG agent that can
route questions across the ECU-700 and ECU-800 Markdown specifications, return
answers with sources, validate against the supplied golden questions, and run
locally with optional DeepSeek LLM generation and local FAISS retrieval.

## Architecture

The runtime flow is:

`User query -> route_query -> retrieve -> plan tool calls -> execute tools -> grounded synthesis -> validate -> answer`

- `documents.py` loads the three Markdown documents and chunks by headings and
  paragraphs. It intentionally does not depend on strict Markdown table parsing,
  because the ECU-700 CAN row is malformed in the source file.
- `retriever.py` uses one retrieval path: `sentence-transformers` creates
  normalized embeddings and FAISS performs in-memory inner-product similarity
  search over the document chunks.
- `graph.py` builds the four-node LangGraph workflow when `langgraph` is
  installed. The same node functions run locally without LangGraph so the
  package remains testable in lightweight environments.
- `answering.py` implements a Plan-Execute style agent controller. When
  `ME_USE_LLM_PLANNER=true`, the configured chat model chooses function calls
  instead of answering directly. The offline fallback planner uses the same tool
  interface so local tests remain deterministic.
- `tools.py` exposes the internal functions available to the agent:
  `search_documents`, `read_model_spec`, `compare_model_specs`, and
  `list_models`.
- `mcp_server.py` runs a real Model Context Protocol server using the official
  Python SDK. MCP clients can discover and call the same ECU tools over stdio or
  Streamable HTTP.
- `llm.py` can use `DEEPSEEK_API_KEY` for planning and/or final synthesis, but
  final answers are constrained to tool evidence and source filenames.
- `model.py` wraps the agent as an `mlflow.pyfunc.PythonModel` with a `predict`
  method that accepts strings, dictionaries, lists, or dataframe-like inputs.

The anti-hallucination guardrails are intentionally simple and inspectable:

- the LLM planner may only return tool calls, not final facts;
- the final answer is generated from returned tool evidence;
- answers include source filenames from the internal documentation;
- numeric values and commands are checked against tool evidence;
- missing evidence or unsupported ECU models reduce confidence and can trigger
  the human review queue.

## Local Usage

Run the assistant directly from the source tree:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant "Compare the CAN bus capabilities of ECU-750 and ECU-850."
```

Run the golden evaluation:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant.evaluate
```

Run the unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The local retriever requires `sentence-transformers` and `faiss-cpu`. For local
LLM configuration, copy `.env.example` to `.env` and fill in your own key:

```bash
cp .env.example .env
$EDITOR .env
```

Do not commit `.env`. It is ignored by git.

DeepSeek is used for chat/completion calls here. Embeddings are local via
`sentence-transformers` + FAISS because this project does not rely on a DeepSeek
embedding model.

DeepSeek can be used as the planning model, the final synthesis model, or both.
For the most agentic local demo, enable planner and grounded answer synthesis:

```bash
ME_USE_LLM_PLANNER=true ME_USE_LLM_ANSWER=true PYTHONPATH=src python3 -m me_engineering_assistant \
  "Summarize the differences between ECU-850 and ECU-850b."
```

For deterministic offline tests, leave those flags as `false`; the same
function-call tools are still used through the local fallback planner.

To inspect the agent process, enable trace output:

```bash
PYTHONPATH=src python -m me_engineering_assistant \
  --trace \
  --trace-format markdown \
  "Compare the CAN bus capabilities of ECU-750 and ECU-850."
```

Supported trace formats are `json`, `markdown`, and `mermaid`. The trace shows
the route decision, retrieved sources, generated plan, executed tools, synthesis
mode, grounding checks, and final validation status. To save the visualization:

```bash
PYTHONPATH=src python -m me_engineering_assistant \
  --trace-file agent_trace.md \
  "How do you enable the NPU on the ECU-850b?"
```

## Structured Logs

Every agent answer can be written as a structured JSONL event for human or AI
monitoring. The default path is `logs/agent_events.jsonl` and can be configured
in `.env`:

```bash
ME_AGENT_LOG_ENABLED=true
ME_AGENT_LOG_PATH=logs/agent_events.jsonl
ME_AGENT_LOG_INCLUDE_TRACE=true
ME_AGENT_LOG_LOW_CONFIDENCE_THRESHOLD=0.75
ME_AGENT_LOG_SLOW_SECONDS=10
```

Each record includes the query, answer, sources, confidence, route decision,
retriever backend, latency, review status, optional trace, and detector flags:

- `low_confidence`
- `missing_sources`
- `needs_review`
- `slow_response`

Inspect the log summary:

```bash
python -m me_engineering_assistant.observability --json
```

or use the installed command:

```bash
me_logs --json
```

## MCP Server

Install the MCP extra:

```bash
python -m pip install -e ".[mcp]"
```

Run the server over stdio, which is the usual local MCP transport:

```bash
PYTHONPATH=src python -m me_engineering_assistant.mcp_server \
  --docs-dir .
```

The server exposes these MCP tools:

- `search_documents`: retrieve source chunks from ECU manuals;
- `read_model_spec`: return extracted specs for one ECU model;
- `compare_model_specs`: compare fields across models;
- `list_models`: list indexed ECU models.

It also exposes resources:

- `ecu://models`
- `ecu://docs/{source}`

For HTTP-based MCP clients, run Streamable HTTP:

```bash
PYTHONPATH=src python -m me_engineering_assistant.mcp_server \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000
```

Then connect the MCP client to `http://127.0.0.1:8000/mcp`.

Client configuration examples are included in:

- `mcp-client-config.json` for clients that accept an `mcpServers` JSON block;
- `.cursor/mcp.json` for Cursor-style project-local MCP configuration.

Install local development dependencies with:

```bash
python -m pip install -e ".[dev,mcp]"
```

The package targets Python 3.10-3.13 because the local FAISS and
sentence-transformers stack is the only supported retrieval path.

For the full Databricks/LangChain/MLflow environment, install:

```bash
python -m pip install -e ".[databricks,dev]"
```

## Databricks Deployment

Configure the bundle target and endpoint variables, then deploy and run:

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run me_engineering_assistant_build_and_log -t dev \
  --var llm_endpoint=<your-llm-serving-endpoint>
```

If your workspace uses a different cloud node type, override `node_type_id`:

```bash
databricks bundle run me_engineering_assistant_build_and_log -t dev \
  --var node_type_id=<workspace-node-type>
```

The job first validates the agent against `test-questions.csv` with a minimum
accuracy threshold of 0.8, then logs an MLflow pyfunc model and records golden
accuracy, average latency, pass count, and the detailed evaluation report.

## Validation Strategy

The supplied `test-questions.csv` is used as a golden regression suite. The
validation checks:

- single-source factual answers for ECU-750, ECU-850, and ECU-850b;
- cross-document comparisons for CAN bus, storage, and temperature range;
- feature availability, including OTA support and negative confirmation;
- exact command retrieval for the ECU-850b NPU enablement command;
- answer accuracy, source match rate, route match rate, confidence, review rate,
  and response latency metrics.

Run the local evaluation framework:

```bash
python -m me_engineering_assistant.evaluate --output-json evaluation_report.json
```

Log the same custom metrics to a local MLflow experiment:

```bash
python -m pip install -e ".[eval]"
python -m me_engineering_assistant.evaluate \
  --log-mlflow \
  --experiment-name me-engineering-assistant-local \
  --run-name golden-evaluation
```

The low-confidence human review mechanism is controlled by `.env`:

```bash
ME_HITL_ENABLED=true
ME_HITL_CONFIDENCE_THRESHOLD=0.75
ME_REVIEW_QUEUE_PATH=review_queue.jsonl
```

When an answer confidence is below the threshold, the agent appends a structured
JSONL record with the query, draft answer, route, sources, retrieved excerpts,
and review reason. This creates a practical SME review queue without requiring
Databricks or an external ticketing system.

For production, extend this with SME-reviewed examples, refusal checks for
unsupported models, and MLflow Evaluation runs against each newly indexed
documentation release.

## Limitations and Future Work

- The current corpus is tiny, so an in-memory vector store is sufficient.
- Router rules are tuned to the ECU-750, ECU-850, and ECU-850b naming scheme.
- Large-scale use should replace the in-memory store with a persistent vector
  index, add incremental re-indexing, and introduce document version metadata.
- Human review can be added for low-confidence answers or queries that mention
  models outside the indexed documentation.
