from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from me_engineering_assistant.model import ECUAssistantPyFunc


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ME_USE_LLM_PLANNER"] = "false"
        os.environ["ME_USE_LLM_ANSWER"] = "false"
        os.environ["ME_FORCE_LLM"] = "false"

    def test_predict_accepts_single_string(self) -> None:
        model = ECUAssistantPyFunc(docs_dir=str(ROOT))
        result = model.predict(None, "How much RAM does the ECU-850 have?")

        self.assertIn("2 GB", result["answer"])
        self.assertIn("ECU-800_Series_Base.md", result["sources"])

    def test_predict_accepts_batch_dictionary(self) -> None:
        model = ECUAssistantPyFunc(docs_dir=str(ROOT))
        result = model.predict(
            None,
            {
                "query": [
                    "What is the power consumption of the ECU-850b under load?",
                    "Which ECU models support Over-the-Air updates?",
                ]
            },
        )

        self.assertEqual(len(result), 2)
        self.assertIn("1.7A", result[0]["answer"])
        self.assertIn("ECU-750", result[1]["answer"])


if __name__ == "__main__":
    unittest.main()
