from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant import answering
from me_engineering_assistant.answering import infer_query_intent
from me_engineering_assistant.graph import ECUAgent


class AnsweringTests(unittest.TestCase):
    def test_answering_has_no_question_specific_synthesizers(self) -> None:
        synthesizers = [
            name
            for name, _function in inspect.getmembers(answering, inspect.isfunction)
            if name.startswith("synthesize_")
        ]

        self.assertEqual(synthesizers, [])

    def test_intent_does_not_treat_modal_can_as_can_bus(self) -> None:
        intent = infer_query_intent("Which ECU can operate in the harshest temperature conditions?")

        self.assertEqual(intent.operation, "rank")
        self.assertEqual(intent.fields, ["operating_temperature"])

    def test_generic_composer_answers_unseen_field_combination(self) -> None:
        agent = ECUAgent(docs_dir=ROOT)

        response = agent.answer("Tell me the processor and storage for ECU-850b.")

        self.assertIn("Dual-core ARM Cortex-A53 @ 1.5 GHz", response.answer)
        self.assertIn("32 GB eMMC", response.answer)


if __name__ == "__main__":
    unittest.main()
