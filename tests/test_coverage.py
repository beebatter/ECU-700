from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.answering import corrective_retrieval
from me_engineering_assistant.coverage import check_plan_coverage
from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.planner import QueryPlan


def _storage_plan() -> QueryPlan:
    return QueryPlan(
        task_type="retrieval_plan",
        entities=["ECU-750", "ECU-850", "ECU-850b"],
        attribute="storage",
        scope="all_indexed_models",
        route="standard",
        subqueries=[
            {"entity": "ECU-750", "attribute": "storage", "query": "ECU-750 storage"},
            {"entity": "ECU-850", "attribute": "storage", "query": "ECU-850 storage"},
            {"entity": "ECU-850b", "attribute": "storage", "query": "ECU-850b storage"},
        ],
        reasons=["test_plan"],
    )


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ME_USE_LLM_PLANNER"] = "false"
        os.environ["ME_USE_LLM_ANSWER"] = "false"
        os.environ["ME_FORCE_LLM"] = "false"
        cls.agent = ECUAgent(docs_dir=ROOT)

    def test_q8_storage_coverage_is_complete(self) -> None:
        plan = _storage_plan()
        evidence = self.agent.retriever.retrieve(
            "storage capacity across all ECU models",
            filters={"models": plan.entities, "fields": [plan.attribute]},
            top_k=10,
        )
        report = check_plan_coverage(plan, evidence)

        self.assertTrue(report.complete)
        self.assertEqual(len(report.items), 3)

    def test_corrective_retrieval_fills_missing_storage_evidence(self) -> None:
        plan = _storage_plan()
        partial = self.agent.retriever.retrieve(
            "storage capacity across all ECU models",
            filters={"models": ["ECU-850", "ECU-850b"], "fields": ["storage"]},
            top_k=10,
        )
        report = check_plan_coverage(plan, partial)

        corrections = corrective_retrieval(toolbox=self.agent.toolbox, query_plan=plan, missing=report.missing)
        combined = partial + [
            result
            for correction in corrections
            for result in self.agent.retriever.retrieve(
                correction.arguments["query"],
                filters={"models": correction.arguments["models"], "fields": correction.arguments["fields"]},
                top_k=3,
            )
        ]
        repaired = check_plan_coverage(plan, combined)

        self.assertFalse(report.complete)
        self.assertTrue(repaired.complete)

    def test_incomplete_coverage_lowers_confidence(self) -> None:
        response = self.agent.answer("How much storage does ECU-999 have?")

        self.assertLess(response.confidence, 0.75)
        self.assertTrue(response.needs_review)


if __name__ == "__main__":
    unittest.main()
