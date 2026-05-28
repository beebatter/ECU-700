# Test Report

## Summary

- Total: 24
- Passed: 24
- Failed: 0
- Errors: 0
- Skipped: 0
- Pass rate: 100.00%
- Duration: 105.70s
- Return code: 0

## Command

```bash
/Users/easkwon/Downloads/ECU-700/.venv/bin/python -m pytest tests --junitxml /Users/easkwon/Downloads/ECU-700/test_reports/junit.xml -q
```

## Test Cases

| Status | Test | Time |
|---|---|---:|
| passed | `tests.test_agent.AgentTests.test_answers_key_acceptance_questions` | 87.136s |
| passed | `tests.test_agent.AgentTests.test_golden_evaluation_passes_acceptance_threshold` | 0.161s |
| passed | `tests.test_agent.AgentTests.test_routes_comparison_queries_to_multiple_sources` | 0.000s |
| passed | `tests.test_agent.AgentTests.test_routes_single_model_queries` | 0.000s |
| passed | `tests.test_answering.AnsweringTests.test_answering_has_no_field_intent_helpers` | 0.070s |
| passed | `tests.test_answering.AnsweringTests.test_answering_has_no_question_specific_synthesizers` | 0.000s |
| passed | `tests.test_answering.AnsweringTests.test_open_ended_question_uses_retrieved_evidence` | 0.024s |
| passed | `tests.test_answering.AnsweringTests.test_standard_rag_answers_unseen_field_combination_from_evidence` | 0.008s |
| passed | `tests.test_answering.AnsweringTests.test_subjective_unbacked_question_goes_to_review` | 0.039s |
| passed | `tests.test_cli.CLITests.test_format_response_is_plain_text_not_raw_json` | 0.000s |
| passed | `tests.test_documents.DocumentTests.test_chunks_do_not_require_strict_markdown_tables` | 0.000s |
| passed | `tests.test_documents.DocumentTests.test_loads_expected_documents_with_metadata` | 0.000s |
| passed | `tests.test_documents.DocumentTests.test_table_text_remains_available_for_rag_retrieval` | 0.000s |
| passed | `tests.test_mcp_server.MCPServerTests.test_stdio_server_lists_and_calls_tools` | 15.853s |
| passed | `tests.test_model.ModelTests.test_predict_accepts_batch_dictionary` | 0.144s |
| passed | `tests.test_model.ModelTests.test_predict_accepts_single_string` | 0.066s |
| passed | `tests.test_observability.ObservabilityTests.test_agent_writes_structured_jsonl_log` | 0.067s |
| passed | `tests.test_observability.ObservabilityTests.test_log_summary_is_ai_readable` | 0.000s |
| passed | `tests.test_review.ReviewQueueTests.test_low_confidence_query_is_queued_for_review` | 0.066s |
| passed | `tests.test_tools.ToolTests.test_list_sources_returns_indexed_documents` | 0.058s |
| passed | `tests.test_tools.ToolTests.test_search_documents_returns_grounded_source_chunks` | 0.005s |
| passed | `tests.test_tools.ToolTests.test_tool_manifest_exposes_function_calls` | 0.000s |
| passed | `tests.test_trace.TraceTests.test_agent_response_can_include_plan_trace` | 0.071s |
| passed | `tests.test_trace.TraceTests.test_trace_visualizations_render` | 0.008s |