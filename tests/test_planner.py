from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant import planner
from me_engineering_assistant.documents import (
    build_document_catalog,
    build_model_field_table,
    chunk_documents,
    load_source_documents,
)
from me_engineering_assistant.planner import QueryPlan, plan_query, sanitize_query_plan


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        docs = load_source_documents(base_path=ROOT)
        cls.documents = docs
        cls.catalog = build_document_catalog(docs)
        cls.field_table = build_model_field_table(chunk_documents(docs))

    def test_fallback_plan_does_not_classify_question_intent(self) -> None:
        plan = plan_query(
            "How does the storage capacity compare across all ECU models?",
            llm_enabled=False,
            catalog=self.catalog,
            field_table=self.field_table,
        )

        self.assertEqual(plan.task_type, "rag")
        self.assertIsNone(plan.attribute)
        self.assertEqual(plan.entities, [])
        self.assertEqual(plan.route, "standard")

    def test_explicit_model_mentions_are_linked_as_metadata_entities(self) -> None:
        plan = plan_query("Compare ECU-750 and ECU-850 CAN bus details.", llm_enabled=False, catalog=self.catalog)

        self.assertEqual(plan.entities, ["ECU-750", "ECU-850"])
        self.assertEqual(plan.scope, "multi_model_filtered")

    def test_llm_plan_shape_is_sanitized_without_fixed_task_categories(self) -> None:
        plan = sanitize_query_plan(
            QueryPlan(
                task_type="made up",
                entities=["ECU-850", "ECU-999"],
                attribute="Storage Capacity!",
                scope="bad scope",
                route="unsafe route",
                subqueries=[{"entity": "ECU-850", "query": "ECU-850 storage"}],
                reasons=["test"],
            ),
            catalog=self.catalog,
            field_table=self.field_table,
        )

        self.assertEqual(plan.task_type, "made_up")
        self.assertEqual(plan.entities, ["ECU-850"])
        self.assertEqual(plan.attribute, "storage")
        self.assertEqual(plan.route, "unsafe_route")

    def test_unified_llm_plan_carries_sanitized_tool_calls(self) -> None:
        original = planner.json_with_configured_llm

        def fake_json(_messages):
            return {
                "task_type": "field comparison",
                "entities": ["ECU-850", "ECU-850b", "ECU-999"],
                "attribute": "Storage Capacity!",
                "scope": "multi model",
                "route": "tool planned",
                "subqueries": [{"entity": "ECU-850", "query": "ECU-850 storage"}],
                "reasons": ["test"],
                "tool_calls": [
                    {
                        "name": "search_documents",
                        "arguments": {
                            "query": "compare storage",
                            "models": ["ECU-999"],
                            "top_k": 99,
                            "unexpected": "ignored",
                        },
                    },
                    {
                        "name": "check_evidence_coverage",
                        "arguments": {"models": ["ECU-850", "ECU-850b"], "field": "storage"},
                    },
                    {"name": "delete_everything", "arguments": {}},
                ],
                "requires_coverage": True,
            }

        manifest = [
            {
                "name": "search_documents",
                "input_schema": {"properties": {"query": {}, "models": {}, "top_k": {}}},
            },
            {
                "name": "check_evidence_coverage",
                "input_schema": {"properties": {"models": {}, "field": {}}},
            },
        ]
        try:
            planner.json_with_configured_llm = fake_json
            plan = plan_query(
                "Compare storage across ECU-850 and ECU-850b.",
                llm_enabled=True,
                catalog=self.catalog,
                field_table=self.field_table,
                tool_manifest=manifest,
            )
        finally:
            planner.json_with_configured_llm = original

        self.assertEqual(plan.entities, ["ECU-850", "ECU-850b"])
        self.assertEqual(plan.attribute, "storage")
        self.assertTrue(plan.requires_coverage)
        self.assertEqual([call["name"] for call in plan.tool_calls], ["search_documents", "check_evidence_coverage"])
        self.assertEqual(plan.tool_calls[0]["arguments"]["models"], ["ECU-850", "ECU-850b"])
        self.assertEqual(plan.tool_calls[0]["arguments"]["top_k"], 10)

    def test_planner_has_no_keyword_intent_helpers(self) -> None:
        helper_names = {name for name, _function in inspect.getmembers(planner, inspect.isfunction)}

        self.assertNotIn("infer_attribute", helper_names)
        self.assertNotIn("_is_comparison_query", helper_names)
        self.assertNotIn("_is_subjective_query", helper_names)


if __name__ == "__main__":
    unittest.main()
