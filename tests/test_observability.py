from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.observability import load_log_records, summarize_log_records


class ObservabilityTests(unittest.TestCase):
    def test_agent_writes_structured_jsonl_log(self) -> None:
        keys = (
            "ME_AGENT_LOG_ENABLED",
            "ME_AGENT_LOG_PATH",
            "ME_AGENT_LOG_INCLUDE_TRACE",
            "ME_USE_LLM_PLANNER",
            "ME_USE_LLM_ANSWER",
            "ME_FORCE_LLM",
        )
        previous = {key: os.environ.get(key) for key in keys}
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "agent_events.jsonl"
            try:
                os.environ["ME_AGENT_LOG_ENABLED"] = "true"
                os.environ["ME_AGENT_LOG_PATH"] = str(log_path)
                os.environ["ME_AGENT_LOG_INCLUDE_TRACE"] = "true"
                os.environ["ME_USE_LLM_PLANNER"] = "false"
                os.environ["ME_USE_LLM_ANSWER"] = "false"
                os.environ["ME_FORCE_LLM"] = "false"

                agent = ECUAgent(docs_dir=ROOT)
                response = agent.answer("How much RAM does the ECU-850 have?")
                records = load_log_records(log_path)

                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["event_type"], "agent_response")
                self.assertEqual(records[0]["query"], "How much RAM does the ECU-850 have?")
                self.assertEqual(records[0]["sources"], response.sources)
                self.assertEqual(records[0]["retriever_backend"], "sentence-transformers-faiss")
                self.assertIn("trace", records[0])
                self.assertFalse(records[0]["detectors"]["missing_sources"])
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_log_summary_is_ai_readable(self) -> None:
        records = [
            {
                "event_type": "agent_response",
                "confidence": 0.4,
                "latency_seconds": 12,
                "sources": [],
                "needs_review": True,
                "detectors": {
                    "low_confidence": True,
                    "missing_sources": True,
                    "needs_review": True,
                    "slow_response": True,
                },
            }
        ]

        summary = summarize_log_records(records)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["low_confidence_count"], 1)
        self.assertEqual(summary["missing_sources_count"], 1)
        self.assertEqual(summary["needs_review_count"], 1)
        self.assertEqual(summary["slow_response_count"], 1)
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
