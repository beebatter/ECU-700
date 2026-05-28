from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
from me_engineering_assistant.graph import ECUAgent
from me_engineering_assistant.visualization import trace_to_markdown, trace_to_mermaid


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ME_USE_LLM_PLANNER"] = "false"
        os.environ["ME_USE_LLM_ANSWER"] = "false"
        os.environ["ME_FORCE_LLM"] = "false"
        cls.agent = ECUAgent(docs_dir=ROOT)

    def test_agent_response_can_include_plan_trace(self) -> None:
        response = self.agent.answer("Compare the CAN bus capabilities of ECU-750 and ECU-850.", include_trace=True)
        payload = response.to_dict()
        step_names = [step["name"] for step in payload["trace"]]
        plan_step = next(step for step in payload["trace"] if step["name"] == "plan")

        self.assertEqual(
            step_names,
            ["route_query", "retrieve", "plan", "execute_tools", "coverage_check", "synthesize", "grounding", "validate"],
        )
        self.assertIn("search_documents", str(plan_step["details"]["tool_calls"]))

    def test_trace_visualizations_render(self) -> None:
        payload = self.agent.answer("How much RAM does the ECU-850 have?", include_trace=True).to_dict()

        self.assertIn("ECU Assistant Trace", trace_to_markdown(payload))
        self.assertIn("flowchart LR", trace_to_mermaid(payload["trace"]))


if __name__ == "__main__":
    unittest.main()
