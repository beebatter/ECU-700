from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant import planner
from me_engineering_assistant.documents import build_model_field_table, chunk_documents, load_source_documents
from me_engineering_assistant.planner import QueryPlan, plan_query, sanitize_query_plan


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        docs = load_source_documents(base_path=ROOT)
        cls.documents = docs
        cls.field_table = build_model_field_table(chunk_documents(docs))

    def test_fallback_plan_does_not_classify_question_intent(self) -> None:
        plan = plan_query(
            "How does the storage capacity compare across all ECU models?",
            llm_enabled=False,
            field_table=self.field_table,
        )

        self.assertEqual(plan.task_type, "rag")
        self.assertIsNone(plan.attribute)
        self.assertEqual(plan.entities, [])
        self.assertEqual(plan.route, "standard")

    def test_explicit_model_mentions_are_linked_as_metadata_entities(self) -> None:
        plan = plan_query("Compare ECU-750 and ECU-850 CAN bus details.", llm_enabled=False)

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
            field_table=self.field_table,
        )

        self.assertEqual(plan.task_type, "made_up")
        self.assertEqual(plan.entities, ["ECU-850"])
        self.assertEqual(plan.attribute, "storage")
        self.assertEqual(plan.route, "unsafe_route")

    def test_planner_has_no_keyword_intent_helpers(self) -> None:
        helper_names = {name for name, _function in inspect.getmembers(planner, inspect.isfunction)}

        self.assertNotIn("infer_attribute", helper_names)
        self.assertNotIn("_is_comparison_query", helper_names)
        self.assertNotIn("_is_subjective_query", helper_names)


if __name__ == "__main__":
    unittest.main()
