# ME Engineering Assistant Agent

This project implements the ECU engineering coding challenge as a small Python
package instead of a notebook. It provides a LangGraph-style RAG agent that can
route questions across the ECU-700 and ECU-800 Markdown specifications, return
answers with sources, validate against the supplied golden questions, and run
locally with Databricks-first or DeepSeek-compatible LLM generation plus local
hybrid retrieval.

## Architecture

The runtime flow is hybrid RAG with optional application-layer memory:

`User query -> load_memory -> unified LLM query/tool plan -> hybrid retrieve -> optional coverage check -> grounded synthesis -> validate -> reflect_memory -> answer`

- `documents.py` loads the three Markdown documents and chunks by headings and
  paragraphs, then creates field-level chunks from table/spec rows. Field
  metadata is dynamic and comes from source labels such as `Storage`, not from
  golden-test question handlers.
- `retriever.py` uses one hybrid retrieval path: `sentence-transformers` creates
  normalized embeddings, FAISS performs dense similarity search, `rank-bm25`
  provides sparse keyword scoring, and a local reranker combines dense, sparse,
  metadata, and field matches.
- `planner.py` produces a structured retrieval and tool-use plan without
  hard-coded intent categories. With an LLM enabled, one planning call proposes
  entities, optional fields, subqueries, tool calls, and whether model-field
  coverage is required. Without an LLM, the fallback only links explicit ECU
  model/series mentions to metadata and leaves the query broad for retrieval.
- `coverage.py` checks requested model-field pairs only when the LLM plan asks
  for coverage, then can trigger corrective retrieval before answer generation.
- `graph.py` builds the LangGraph workflow when `langgraph` is installed. The
  same node functions run locally without LangGraph so the package remains
  testable in lightweight environments.
- `memory.py`, `conversation.py`, and `reflection.py` add optional multi-turn
  memory outside the retrieval core. Short-term memory is recent turns,
  mid-term memory is a per-session summary, and long-term memory is a small
  SQLite table of stable preferences or project decisions. Reflection redacts
  secrets before anything is persisted.
- `answering.py` implements the RAG answer controller. It does not maintain a
  growing list of intent handlers, specification-specific synthesizers, or
  one function per expected user question. Every query follows the same rule:
  execute the query plan, retrieve evidence, verify coverage when the plan names
  a field, compose only from returned evidence, and rely on grounded LLM
  synthesis for open-ended intent judgment when enabled.
- `tools.py` exposes the internal functions available to the agent:
  `search_documents`, `get_document_catalog`, `get_model_field_evidence`,
  `check_evidence_coverage`, and `list_sources`.
- `mcp_server.py` runs a real Model Context Protocol server using the official
  Python SDK. MCP clients can discover and call the same ECU tools over stdio or
  Streamable HTTP.
- `llm.py` uses `DATABRICKS_LLM_ENDPOINT` first and falls back to
  `DEEPSEEK_API_KEY` for local development. Final answers are constrained to
  tool evidence and source filenames.
- `model.py` wraps the agent as an `mlflow.pyfunc.PythonModel` with a `predict`
  method that accepts strings, dictionaries, lists, or dataframe-like inputs.
  Requests may include `session_id` when memory is enabled.

The anti-hallucination guardrails are intentionally simple and inspectable:

- the LLM planner may only create a structured retrieval/tool plan, not final facts;
- the final answer is generated from returned tool evidence;
- entity-field coverage is checked when the LLM plan explicitly requests it;
- answers include source filenames from the internal documentation;
- numeric values and commands are checked against tool evidence;
- unsupported criteria are handled by grounded LLM synthesis when enabled, and
  unsupported models reduce confidence in deterministic fallback mode;
- missing evidence or unsupported ECU models reduce confidence and can trigger
  the human review queue.

## Local Usage

Run the assistant directly from the source tree:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant "Compare the CAN bus capabilities of ECU-750 and ECU-850."
```

Start the minimal interactive CLI:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant
```

The prompt asks `Question>`; type a question and the assistant prints separated
`Answer`, `Sources`, and `Confidence` sections instead of raw JSON.

Interactive CLI memory is enabled by default. The default session id is
`default`, so follow-up questions can use the recent conversation and future CLI
runs can reuse the same session summary:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant --session-id ecu-demo
```

Memory is stored in SQLite at `.me_engineering_memory.sqlite` unless overridden:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant \
  --session-id ecu-demo \
  --memory-db .local/ecu-memory.sqlite
```

Disable memory for a stateless run:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant --no-memory "How much RAM does ECU-850 have?"
```

Run the golden evaluation:

```bash
PYTHONPATH=src python3 -m me_engineering_assistant.evaluate
```

Run the unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The local retriever requires `sentence-transformers`, `faiss-cpu`, and
`rank-bm25`. For local LLM configuration, copy `.env.example` to `.env` and
fill in your own key:

```bash
cp .env.example .env
$EDITOR .env
```

Do not commit `.env`. It is ignored by git.

DeepSeek is used for chat/completion calls here. Embeddings and sparse retrieval
are local via `sentence-transformers` + FAISS + BM25 because this project does
not rely on a DeepSeek embedding model.

Databricks is the preferred LLM path when `DATABRICKS_LLM_ENDPOINT` is set.
DeepSeek is the local development fallback when `DEEPSEEK_API_KEY` is set. The
sample configuration enables unified LLM query/tool planning and grounded LLM
answer synthesis:

```bash
ME_USE_LLM_PLANNER=true ME_USE_LLM_TOOL_PLANNER=true ME_USE_LLM_ANSWER=true ME_FORCE_LLM=true \
PYTHONPATH=src python3 -m me_engineering_assistant \
  "Summarize the differences between ECU-850 and ECU-850b."
```

For deterministic offline tests, set those flags to `false`; the same RAG
retrieval tool is still used through the local fallback planner, but no
question-type or field-keyword classifier is used. `ME_USE_LLM_TOOL_PLANNER` is
kept for backward-compatible configuration but the current runtime uses one
unified planning call rather than a second tool-planning LLM call.

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
- `get_document_catalog`: list indexed source documents with model metadata;
- `get_model_field_evidence`: inspect structured model-field evidence;
- `check_evidence_coverage`: verify model-field coverage before answering;
- `list_sources`: list indexed ECU source documents.

It also exposes resources:

- `ecu://sources`
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

The package targets Python 3.10-3.13 because the local FAISS,
sentence-transformers, and BM25 stack is the only supported retrieval path.

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
  total latency, LLM call count, LLM latency, retrieval latency, and generation
  latency metrics.

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
- Entity linking currently recognizes explicit ECU model/series identifiers from
  the indexed catalog. Open-ended intent understanding should use the LLM
  planner and grounded answer synthesis.
- Large-scale use should replace the in-memory store with a persistent vector
  index, add incremental re-indexing, and introduce document version metadata.
- Human review can be added for low-confidence answers or queries that mention
  models outside the indexed documentation.
