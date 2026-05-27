from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from me_engineering_assistant.evaluate import evaluate_golden, load_questions
from me_engineering_assistant.graph import ECUAgent, route_query


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ME_USE_LLM_PLANNER"] = "false"
        os.environ["ME_USE_LLM_ANSWER"] = "false"
        os.environ["ME_FORCE_LLM"] = "false"
        os.environ["ME_RETRIEVER_BACKEND"] = "keyword"
        cls.agent = ECUAgent(docs_dir=ROOT, prefer_langchain=False)

    def test_routes_single_model_queries(self) -> None:
        self.assertEqual(route_query("How much RAM does the ECU-850 have?").models, ["ECU-850"])
        self.assertEqual(route_query("What are the AI capabilities of the ECU-850b?").models, ["ECU-850b"])
        self.assertEqual(route_query("What is the max temp for ECU-750?").models, ["ECU-750"])

    def test_routes_comparison_queries_to_multiple_sources(self) -> None:
        route = route_query("Compare the CAN bus capabilities of ECU-750 and ECU-850.")

        self.assertEqual(route.mode, "multi_source")
        self.assertEqual(route.models, ["ECU-750", "ECU-850"])

    def test_answers_key_acceptance_questions(self) -> None:
        temp = self.agent.answer("What is the maximum operating temperature for the ECU-750?")
        can = self.agent.answer("Compare the CAN bus capabilities of ECU-750 and ECU-850.")
        command = self.agent.answer("How do you enable the NPU on the ECU-850b?")

        self.assertIn("+85", temp.answer)
        self.assertIn("1 Mbps", can.answer)
        self.assertIn("2 Mbps", can.answer)
        self.assertIn("me-driver-ctl --enable-npu --mode=performance", command.answer)

    def test_golden_evaluation_passes_acceptance_threshold(self) -> None:
        report = evaluate_golden(self.agent, load_questions(ROOT / "test-questions.csv"))

        self.assertEqual(report["total"], 10)
        self.assertGreaterEqual(report["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
