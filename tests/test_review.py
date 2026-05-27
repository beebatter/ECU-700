from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from me_engineering_assistant.graph import ECUAgent


class ReviewQueueTests(unittest.TestCase):
    def test_low_confidence_query_is_queued_for_review(self) -> None:
        keys = (
            "DEEPSEEK_API_KEY",
            "ME_HITL_ENABLED",
            "ME_HITL_CONFIDENCE_THRESHOLD",
            "ME_REVIEW_QUEUE_PATH",
            "ME_USE_LLM_PLANNER",
            "ME_USE_LLM_ANSWER",
            "ME_FORCE_LLM",
        )
        previous = {key: os.environ.get(key) for key in keys}
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "review_queue.jsonl"
            try:
                os.environ["DEEPSEEK_API_KEY"] = ""
                os.environ["ME_HITL_ENABLED"] = "true"
                os.environ["ME_HITL_CONFIDENCE_THRESHOLD"] = "0.75"
                os.environ["ME_REVIEW_QUEUE_PATH"] = str(queue_path)
                os.environ["ME_USE_LLM_PLANNER"] = "false"
                os.environ["ME_USE_LLM_ANSWER"] = "false"
                os.environ["ME_FORCE_LLM"] = "false"

                agent = ECUAgent(docs_dir=ROOT)
                response = agent.answer("What is the LIN bus termination value for ECU-999?")

                self.assertTrue(response.needs_review)
                self.assertIsNotNone(response.review_id)
                record = json.loads(queue_path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(record["review_id"], response.review_id)
                self.assertIn("confidence", record["reason"])
                self.assertEqual(record["query"], "What is the LIN bus termination value for ECU-999?")
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
