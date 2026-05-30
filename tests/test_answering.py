from __future__ import annotations

import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant import answering
from me_engineering_assistant.graph import ECUAgent


class AnsweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ME_USE_LLM_PLANNER"] = "false"
        os.environ["ME_USE_LLM_ANSWER"] = "false"
        os.environ["ME_FORCE_LLM"] = "false"
        os.environ["ME_AGENT_LOG_ENABLED"] = "false"
        cls.agent = ECUAgent(docs_dir=ROOT)

    def test_answering_has_no_question_specific_synthesizers(self) -> None:
        synthesizers = [
            name
            for name, _function in inspect.getmembers(answering, inspect.isfunction)
            if name.startswith("synthesize_")
        ]

        self.assertEqual(synthesizers, [])

    def test_answering_has_no_field_intent_helpers(self) -> None:
        helper_names = {name for name, _function in inspect.getmembers(answering, inspect.isfunction)}

        self.assertNotIn("infer_query_intent", helper_names)
        self.assertNotIn("infer_fields_from_query", helper_names)

    def test_standard_rag_answers_unseen_field_combination_from_evidence(self) -> None:
        response = self.agent.answer("Tell me the processor and storage for ECU-850b.")

        self.assertIn("Dual-core ARM Cortex-A53 @ 1.5 GHz", response.answer)
        self.assertIn("32 GB eMMC", response.answer)

    def test_open_ended_question_uses_retrieved_evidence(self) -> None:
        query = "Which is better for edge AI deployment, ECU-850 or ECU-850b?"

        response = self.agent.answer(query)

        self.assertIn("ECU-850b", response.answer)
        self.assertIn("5 TOPS", response.answer)

    def test_grounding_accepts_numeric_unit_spacing_variants(self) -> None:
        evidence = [
            answering.RetrievalResult(
                content="Power Consumption: Idle: 550mA, Under Load: 1.7A",
                metadata={"source": "ECU-800_Series_Plus.md"},
                score=1.0,
            )
        ]

        unsupported = answering.unsupported_numeric_or_command_claims(
            "The ECU-850b uses 1.7 A under load.",
            tool_results=[],
            evidence=evidence,
        )

        self.assertEqual(unsupported, [])

    def test_llm_tool_plan_is_sanitized_before_execution(self) -> None:
        plan = answering.AgentPlan(
            rationale="LLM supplied extra arguments.",
            calls=[
                answering.ToolCall(
                    "search_documents",
                    {
                        "query": "storage capacity",
                        "models": "ECU-850",
                        "series": ["ECU-800"],
                        "unexpected": "ignored",
                        "top_k": 99,
                    },
                )
            ],
        )

        sanitized = answering.sanitize_plan(
            query="How does storage compare?",
            route={"models": ["ECU-750", "ECU-850", "ECU-850b"]},
            toolbox=self.agent.toolbox,
            plan=plan,
        )

        self.assertEqual(len(sanitized.calls), 1)
        arguments = sanitized.calls[0].arguments
        self.assertNotIn("unexpected", arguments)
        self.assertEqual(arguments["models"], ["ECU-850"])
        self.assertEqual(arguments["series"], ["ECU-800"])
        self.assertEqual(arguments["top_k"], 10)

    def test_no_hardcoded_subjective_intent_gate(self) -> None:
        previous = {
            "ME_REVIEW_QUEUE_PATH": os.environ.get("ME_REVIEW_QUEUE_PATH"),
            "ME_HITL_ENABLED": os.environ.get("ME_HITL_ENABLED"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.environ["ME_REVIEW_QUEUE_PATH"] = str(Path(temp_dir) / "review.jsonl")
                os.environ["ME_HITL_ENABLED"] = "true"

                response = self.agent.answer("ECU-850 和 ECU-850b 哪个更好看一点?")

                self.assertIn("Source:", response.answer)
                self.assertFalse(response.needs_review)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
